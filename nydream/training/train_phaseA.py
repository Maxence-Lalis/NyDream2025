import torch
from torch_geometric.loader import DataLoader as pygdl
from torch import nn

import pandas
import os
import tqdm
import numpy as np

from dataloader.dataloader import DreamLoader
from model.gnn.graphnets_film_vector import GraphNets
from model.gnn.predictor import EndToEndModule
from model.gnn.early_stop import EarlyStopping
from model.prediction_head.GLM import SimpleMLP
from training.make_train_epoch import make_train_epoch
from eval.make_eval_epoch import make_eval_epoch
from training.train_utils import _checkpoint_exists, init_film_identity, freeze_params, is_film_param

from loss import get_loss_fn
from metrics import get_metric_fn

def phaseA_load_and_run(hparams):
    """"""
    # ------------------------------------------------------------------
    # Load data and split
    # ------------------------------------------------------------------

    dl = DreamLoader(full_data_path=hparams["phaseA_data_full_train"],
                smiles_column=hparams["phaseA_data_smiles"],
                label_column=hparams["phaseA_data_label"],
                id_column=hparams["phaseA_data_id"],
                conc_column=hparams['phaseA_conc_column'],
                save_dir=hparams["PhaseA_save_dir"],
                sep=hparams["phaseA_data_sep"],
                mode=hparams["mode"],
                train_path=hparams['phaseA_data_train'],
                valid_path=hparams['phaseA_data_valid'],
                test_path=hparams['phaseA_data_test'],
                exclude_path=hparams['phaseA_data_exclude']
                )
    if hparams['phaseA_custom_split']:
        train_ds, valid_ds, test_ds = dl.load_split()
    else:
        train_ds, valid_ds, test_ds = dl.split()

    train_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=hparams['phaseA_init_globals'])
    valid_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=hparams['phaseA_init_globals'])
    dl.save()
    
    print("[training] train set size:", len(train_ds))
    print("[training] valid set size:", len(valid_ds))
    print("[training] test set size:", len(test_ds))

    train_loader = pygdl(train_ds, batch_size=hparams['phaseA_train_batch_size'], shuffle=True)
    valid_loader = pygdl(valid_ds, batch_size=hparams['phaseA_valid_batch_size'], shuffle=False)


    # ------------------------------------------------------------------
    # Build model (infer node/edge dims from a sample molecule)
    # ------------------------------------------------------------------
    
    node_dim = train_ds.features[0].x.size(1) if hasattr(train_ds.features[0], "x") else 50
    edge_dim = train_ds.features[0].edge_attr.size(1) if hasattr(train_ds.features[0], "edge_attr") else 10
    global_dim = train_ds.features[0].u.size(1)
    
    gnn = GraphNets(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=hparams['graphnet_hidden_dims'],
        depth=hparams['graphnet_depth'],
        dropout=hparams['graphnet_dropout'],
    ).to(hparams['PhaseA_device'])
    
    # ------------------------------------------------------------------
    # Initialize weights
    # ------------------------------------------------------------------
    pred = SimpleMLP(input_dim=global_dim, output_dim=hparams['phaseA_data_label_size']).to(hparams['PhaseA_device'])    
    
    with torch.no_grad():
        train_mean = train_ds.labels.mean(axis=0)
        final_lin = [m for m in pred.modules() if isinstance(m, nn.Linear)][-1]      
        final_lin.bias.copy_(torch.tensor(train_mean, dtype=final_lin.bias.dtype))
    
    model = EndToEndModule(gnn, pred).to(hparams['PhaseA_device'])

    if _checkpoint_exists(hparams["PhaseA_save_dir"],
                        "phaseA_gnn_embedder.pt",
                        "phaseA_predictor.pt",
                        load_into=(model.gnn_embedder, model.nn_predictor)):
        print("[training] Phase A checkpoint found - skipping pre-training.")
        return model
    else:
        # ------------------------------------------------------------------
        # Optimize + Early Stopping + Logging 
        # ------------------------------------------------------------------

        gnn.apply(init_film_identity)
        freeze_params(model, is_film_param, freeze=True)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=hparams['phaseA_train_learning_rate'])
        es = EarlyStopping(gnn, patience=hparams['phaseA_early_stop_patience'], mode=hparams['phaseA_mode_patience'], min_delta=hparams['phaseA_early_min_delta'])

        loss_fn = get_loss_fn(hparams['phaseA_loss_fn'])
        metric_fn = get_metric_fn(hparams['phaseA_metric_fn'])

        monitor_scope = hparams['PhaseA_monitor_scope']
        if isinstance(metric_fn, dict):
            monitor_metric = hparams['phaseA_monitor_metric']
            if not monitor_metric or monitor_metric not in metric_fn:
                monitor_metric = next(iter(metric_fn.keys()))
            log_dict = {k: [] for k in ['epoch', 'train_loss', 'val_loss']}
            for name in metric_fn.keys():
                log_dict[f'val_metric_{name}'] = []
        else:
            monitor_metric = None
            log_dict = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric']}
        # ------------------------------------------------------------------
        # Training loop 
        # ------------------------------------------------------------------

        pbar = tqdm.tqdm(range(hparams['phaseA_num_epochs']))
        for epoch in pbar:     
            train_loss = make_train_epoch(model, 
                                        optimizer, train_loader, 
                                        loss_fn, hparams['PhaseA_device']
                                        )
            
            val_loss, val_report = make_eval_epoch(model, 
                                                valid_loader, loss_fn, 
                                                metric_fn, hparams['PhaseA_device']
                                                )

            val_metric = val_report[monitor_scope]['metrics']
            if isinstance(val_metric, dict):
                for name, value in val_metric.items():
                    log_dict[f'val_metric_{name}'].append(value)
            else:
                log_dict['val_metric'].append(val_metric)

            log_dict['epoch'].append(epoch)
            log_dict['train_loss'].append(train_loss)
            log_dict['val_loss'].append(val_loss)

            if monitor_metric:
                pbar.set_description(
                    f"Epoch {epoch} : Train {train_loss:.4f} | Val {val_loss:.4f} | {monitor_metric} {val_metric[monitor_metric]:.4f}"
                )
            else:
                pbar.set_description(
                    f"Epoch {epoch} : Train {train_loss:.4f} | Val {val_loss:.4f} | Metric {val_metric:.4f}"
                )

            # Early stopping based on the chosen metric
            if es.check_criteria(val_metric[monitor_metric], model):
                print(f'[training] Early stop reached at epoch {es.best_step} with metric {es.best_value}')
                break

        # ------------------------------------------------------------------
        # Save best model + Logs 
        # ------------------------------------------------------------------

        best_embedder_dict = es.restore_best()
        model.load_state_dict(best_embedder_dict)

        torch.save(model.gnn_embedder.state_dict(), os.path.join(hparams['PhaseA_save_dir'], 'phaseA_gnn_embedder.pt'))
        torch.save(model.nn_predictor.state_dict(), os.path.join(hparams['PhaseA_save_dir'], 'phaseA_predictor.pt'))

        log_df = pandas.DataFrame(log_dict)
        log_df.to_csv(os.path.join(hparams['PhaseA_save_dir'], 'phaseA_training.csv'), index=False)

        return model