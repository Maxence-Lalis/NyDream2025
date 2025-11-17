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
from training.train_utils import _checkpoint_exists, init_film_high_grad, _set_requires_grad, _build_param_groups, _add_tiny_noise_to_film_scales,is_film_param

from loss import get_loss_fn
from metrics import get_metric_fn

from training.grad_flow import GradFlowMonitor
from collections import OrderedDict



def phaseB_load_and_run(model, hparams):
    """"""
    # ------------------------------------------------------------------
    # Load data and split
    # ------------------------------------------------------------------
    dl = DreamLoader(full_data_path=hparams["PhaseB_data_full_train"],
                smiles_column=hparams["PhaseB_data_smiles"],
                label_column=hparams["PhaseB_data_label"],
                id_column=hparams["PhaseB_data_id"],
                conc_column=hparams['phaseB_conc_column'],
                save_dir=hparams["PhaseB_save_dir"],
                sep=hparams["PhaseB_data_sep"],
                mode=hparams["PhaseB_mode"],
                train_path=hparams['PhaseB_data_train'],
                valid_path=hparams['PhaseB_data_valid'],
                test_path=hparams['PhaseB_data_test'],
                exclude_path=hparams['PhaseB_data_exclude'],
                normalize_labels=False
                )
    
    if hparams['PhaseB_custom_split']:
        train_ds, valid_ds, test_ds = dl.load_split()
    else:
        train_ds, valid_ds, test_ds = dl.split(hparams['phaseB_valid_size'],hparams['phaseB_test_size'],hparams['phaseB_split_strategy'],
                                               hparams['phaseB_dup_test_frac'],hparams['phaseB_dup_n_bins'])

    train_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=False)
    valid_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=False)
    dl.save()

    print("[training] train set size:", len(train_ds))
    print("[training] valid set size:", len(valid_ds))
    print("[training] test set size:", len(test_ds))

    train_loader = pygdl(train_ds, batch_size=hparams['PhaseB_train_batch_size'], shuffle=True)
    valid_loader = pygdl(valid_ds, batch_size=1024, shuffle=False)

    # ------------------------------------------------------------------
    # Initialize weights
    # ------------------------------------------------------------------
    pred = SimpleMLP(input_dim=234, output_dim=hparams['PhaseB_data_label_size']).to(hparams['PhaseB_device'])    
    
    with torch.no_grad():
        final_lin = [m for m in pred.modules() if isinstance(m, nn.Linear)][-1]
        Y = torch.tensor(train_ds.labels, dtype=final_lin.bias.dtype)
        if train_ds._y_transform is not None:
            Y = train_ds._y_transform(Y)
        final_lin.bias.copy_(Y.mean(dim=0))

    model = EndToEndModule(model.gnn_embedder, pred).to(hparams['PhaseB_device'])
    film_grad_monitor = GradFlowMonitor(name_filter_fn=is_film_param).attach(model)
    head_monitor = GradFlowMonitor(lambda n: n.startswith("nn_predictor"))
    head_monitor.attach(model)
    base_monitor = GradFlowMonitor(lambda n: not n.startswith("nn_predictor") and not is_film_param(n)).attach(model)
    if _checkpoint_exists(hparams["PhaseB_save_dir"],
                        "phaseB_gnn_embedder.pt",
                        "phaseB_predictor.pt",
                        load_into=(model.gnn_embedder, model.nn_predictor)):
        print("[training] Phase B checkpoint found - skipping pre-training.")
        return model
    else:
        # ------------------------------------------------------------------
        # Optimize + Early Stopping + Logging 
        # ------------------------------------------------------------------

        # 1) FiLM high-gradient init + tiny stochastic push
        # model.apply(init_film_high_grad)
        # model.apply(lambda m: _add_tiny_noise_to_film_scales(m, std=hparams.get("PhaseB_film_noise_std", 1e-3)))

        # 2) Warmup: train FiLM + head, freeze base
        warmup_epochs = int(hparams.get("PhaseB_warmup_epochs", 8))
        base_lr = float(hparams.get("PhaseB_base_lr", 1e-5))   # for post-warmup base
        head_lr = float(hparams.get("PhaseB_head_lr", 3e-4))
        film_lr = float(hparams.get("PhaseB_film_lr", 3e-4))
        weight_decay = float(hparams.get("PhaseB_weight_decay", 1e-5))

        loss_fn1 = get_loss_fn(hparams['PhaseB_loss_fn1'])
        loss_fn2 = get_loss_fn(hparams['PhaseB_loss_fn2'])
        metric_fn = get_metric_fn(hparams['PhaseB_metric_fn'])

        if isinstance(metric_fn, dict):
            monitor_metric = hparams['monitor_metric'] if hparams.get('monitor_metric') in metric_fn else next(iter(metric_fn.keys()))
            log_dict = {k: [] for k in ['epoch', 'train_loss', 'val_loss']}
            for name in metric_fn.keys():
                log_dict[f'val_metric_{name}'] = []
        else:
            monitor_metric = None
            log_dict = {k: [] for k in ['epoch', 'train_loss', 'val_loss', 'val_metric']}

        es = EarlyStopping(model.gnn_embedder, patience=hparams['PhaseB_early_stop_patience'],
                           mode=hparams['PhaseB_mode_patience'], min_delta=hparams['PhaseB_early_min_delta'])

        # ---- Warmup optimizer: FiLM + head only ----
        _set_requires_grad(model, film=True, head=True, base=False)
        warmup_optimizer = torch.optim.AdamW(
            _build_param_groups(model, base_lr=0.0, head_lr=head_lr, film_lr=film_lr, weight_decay=weight_decay)
        )

        pbar = tqdm.tqdm(range(hparams['PhaseB_num_epochs']))
        for epoch in pbar:
            # switch optimizer after warmup
            if epoch == warmup_epochs:
                # Unfreeze base; rebuild optimizer with 3 param groups
                _set_requires_grad(model, film=True, head=True, base=True)
                optimizer = torch.optim.AdamW(
                    _build_param_groups(model, base_lr=base_lr, head_lr=head_lr, film_lr=film_lr, weight_decay=weight_decay)
                )

            # choose optimizer based on stage
            opt = warmup_optimizer if epoch < warmup_epochs else optimizer

            train_loss = make_train_epoch(
                model, opt, train_loader,
                loss_fn1, hparams['PhaseB_device'], loss_fn2, lambda_val=hparams['PhaseB_lambda_val']
            )

            val_loss, val_metric = make_eval_epoch(
                model, valid_loader, loss_fn1,
                metric_fn, hparams['PhaseB_device'], loss_fn2, hparams['PhaseB_lambda_val']
            )

            film_grad_stats = film_grad_monitor.epoch_stats(reduce_grouped=True, reset=True)
            head_grad_stats = head_monitor.epoch_stats(reduce_grouped=True, reset=True)
            base_monitor_stats = base_monitor.epoch_stats(reduce_grouped=True, reset=True)
            print(film_grad_stats)
            print(head_grad_stats)
            print(base_monitor_stats)
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
                # Early stopping on monitored metric
                if es.check_criteria(val_metric[monitor_metric], model):
                    print(f'[training] Early stop reached at epoch {es.best_step} with metric {es.best_value}')
                    break
            else:
                pbar.set_description(
                    f"Epoch {epoch} : Train {train_loss:.4f} | Val {val_loss:.4f} | Metric {val_metric:.4f}"
                )
                if es.check_criteria(val_metric, model):
                    print(f'[training] Early stop reached at epoch {es.best_step} with metric {es.best_value}')
                    break

        # ------------------------------------------------------------------
        # Save best model + Logs 
        # ------------------------------------------------------------------
        best_embedder_dict = es.restore_best()
        model.load_state_dict(best_embedder_dict)

        torch.save(model.gnn_embedder.state_dict(), os.path.join(hparams['PhaseB_save_dir'], 'phaseB_gnn_embedder.pt'))
        torch.save(model.nn_predictor.state_dict(), os.path.join(hparams['PhaseB_save_dir'], 'phaseB_predictor.pt'))

        log_df = pandas.DataFrame(log_dict)
        log_df.to_csv(os.path.join(hparams['PhaseB_save_dir'], 'phaseB_training.csv'), index=False)

        return model, [train_loader, valid_loader]