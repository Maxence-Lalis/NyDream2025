import torch
from torch_geometric.loader import DataLoader as pygdl
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import pandas
import os
import tqdm
import numpy as np

from dataloader.dataloader import DreamLoader
from model.cVAE.multicVAE import MultiHeadMLP
from model.gnn.early_stop import EarlyStopping
from training.make_train_epoch import make_train_epoch
from eval.make_eval_epoch import make_eval_epoch

from loss import get_loss_fn
from metrics import get_metric_fn
from training.grad_flow import GradFlowMonitor


def phaseC_load_and_run(model, loaders, hparams):
    """"""
    
    # ------------------------------------------------------------------
    # Load data and split
    # ------------------------------------------------------------------
    for p in model.gnn_embedder.parameters():
        p.requires_grad = False
    model.gnn_embedder.eval()

    panel_cfg = hparams['PhaseC_panel']

    dream_train_loader = loaders[0]
    dream_valid_loader = loaders[1]

    embeddings, labels, task_ids, is_train = [], [], [], [] 
    with torch.no_grad():
        for task_idx, (task, cfg) in enumerate(panel_cfg.items()):
            
            if task != "DREAM":
                dl = DreamLoader(full_data_path=cfg["csv"],
                    smiles_column=cfg["smiles_col"],
                    label_column=cfg["label_col"],
                    id_column=cfg["id_col"],
                    conc_column=cfg["conc_col"],
                    save_dir=cfg["save_dir"],
                    sep=cfg["sep"],
                    mode=cfg["mode"],
                    train_path=cfg['train_path'],
                    valid_path=cfg['valid_path'],
                    test_path=cfg['test_path'],
                    exclude_path=cfg['exclude_path'],
                    normalize_labels=cfg['normalize_labels'],
                    norm_method=cfg['norm_method'],
                    )


                if cfg['custom_split']:
                    train_ds, valid_ds, test_ds = dl.load_split()
                else:
                    train_ds, valid_ds, test_ds = dl.split(cfg['valid_size'],cfg['test_size'],cfg['split_strategy'],
                                                        cfg['dup_test_frac'],cfg['dup_n_bins'])
                print(len(train_ds))
                print(len(valid_ds))
                train_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=hparams["PhaseC_init_globals"])
                valid_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=hparams["PhaseC_init_globals"])
                dl.save()
                
                tr_loader = pygdl(train_ds, batch_size=cfg['batch_size'])
                va_loader = pygdl(valid_ds, batch_size=cfg['batch_size'])

                for loader, store in [(tr_loader, True), (va_loader, False)]:
                    for data, y, *extra in loader:
                        h = model.gnn_embedder(data.to(hparams['PhaseC_device'])).detach().cpu()
                        embeddings.append(h)
                        labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
                        task_ids.append(torch.full((h.size(0), 1), task_idx, dtype=torch.long))
                        is_train.append(torch.full((h.size(0), 1), store, dtype=torch.bool))
            else:
                for loader, store in [(dream_train_loader, True), (dream_valid_loader, False)]:
                    for data, y, *extra in loader:
                        h = model.gnn_embedder(data.to(hparams['PhaseC_device'])).detach().cpu()
                        embeddings.append(h)
                        labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
                        task_ids.append(torch.full((h.size(0), 1), task_idx, dtype=torch.long))
                        is_train.append(torch.full((h.size(0), 1), store, dtype=torch.bool))

            print(f"[training] {task} Embedding extraction done.")


    cond_dim = max(cfg["out_dim"] for (task, cfg) in panel_cfg.items())

    labels = [F.pad(t, (0, cond_dim - t.shape[1])) for t in labels]

    X     = torch.cat(embeddings)
    Y     = torch.cat(labels)                  
    TID   = torch.cat(task_ids)
    IS_TR = torch.cat(is_train).squeeze(1)

    num_tasks = len(hparams['PhaseC_panel'])    

    MultiMLP_embed_dim = X.shape[1]
    train_set = TensorDataset(X[IS_TR],  Y[IS_TR],  TID[IS_TR])
    val_set   = TensorDataset(X[~IS_TR], Y[~IS_TR], TID[~IS_TR])

    train_loader    = DataLoader(train_set, batch_size=hparams['PhaceC_batch_size'], shuffle=True)
    valid_loader    = DataLoader(val_set,   batch_size=hparams['PhaceC_batch_size'])

    print("Train set size:", len(train_set))
    print("Test set size:", len(val_set))
    
    # ------------------------------------------------------------------
    # Model + Initialize weights
    # ------------------------------------------------------------------
    
    rata_dims = {t: cfg["out_dim"] for t, cfg in hparams['PhaseC_panel'].items()}
    mMLP = MultiHeadMLP(
        embed_dim  = MultiMLP_embed_dim,
        rata_dims  = rata_dims,
        dropout    = hparams['PhaseC_dropout'],
    ).to(hparams['PhaseC_device'])
    
    head_monitor = GradFlowMonitor(lambda n: n.startswith("heads")).attach(mMLP)
    trunk_monitor = GradFlowMonitor(lambda n: n.startswith("trunk")).attach(mMLP)
    # ------------------------------------------------------------------
    # Optimize + Early Stopping + Logging 
    # ------------------------------------------------------------------
    
    n_by_task = torch.bincount(TID[IS_TR].flatten())
    alpha     = (len(n_by_task) * (1.0 / n_by_task) / (1.0 / n_by_task).sum()).to(hparams['PhaseC_device'])
    tid2task = {i: t for i, t in enumerate(hparams['PhaseC_panel'].keys())}
    # alpha = torch.tensor([1,0.5,0.2], device=device)   # custom weights

    optimizer        = torch.optim.Adam(mMLP.parameters(), lr=hparams['PhaseC_lr'], weight_decay=hparams['PhaseC_weight_decay']) # weight_decay=1e-5
    es  = EarlyStopping(mMLP, patience=hparams['PhaseC_patience'], min_delta=hparams['PhaseC_early_min_delta'], mode="maximize") # patience=200 , mode="maximize"

    loss_fn = get_loss_fn(hparams['PhaseC_loss_fn1'])
    metric_fn = get_metric_fn(hparams['PhaseC_metric_fn'])

    monitor_scope = hparams['PhaseC_monitor_scope']

    if isinstance(metric_fn, dict):
        monitor_metric = hparams['PhaseC_monitor_metric']
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

    pbar = tqdm.tqdm(range(hparams['PhaseC_num_epochs']))
    for epoch in pbar:
        train_loss = make_train_epoch(
            mMLP, optimizer, train_loader, loss_fn, hparams['PhaseC_device'],
            loss_fn2=None, lambda_val=hparams['PhaseC_lambda_val'],
            alpha=alpha, tid2task=tid2task
        )

        val_loss, val_report = make_eval_epoch(
            mMLP, valid_loader, loss_fn, metric_fn, hparams['PhaseC_device'],
            loss_fn2=None, lambda_val=hparams['PhaseC_lambda_val'],
            tid2task=tid2task
        )

        film_grad_stats = trunk_monitor.epoch_stats(reduce_grouped=False, reset=True)
        head_grad_stats = head_monitor.epoch_stats(reduce_grouped=False, reset=True)
        # print(film_grad_stats)
        # print(head_grad_stats)

        val_metric = val_report[monitor_scope]['metrics']
        if isinstance(val_metric, dict):
            for name, value in val_metric.items():
                log_dict[f'val_metric_{name}'].append(value)
        else:
            log_dict['val_metric'].append(val_metric)

        # macro averages
        macro_metrics = val_report['macro_avg']['metrics']
        if isinstance(metric_fn, dict):
            for name, value in macro_metrics.items():
                key = f'val_macro_{name}'
                if key not in log_dict: log_dict[key] = []
                log_dict[key].append(value)
        else:
            key = 'val_macro_metric'
            if key not in log_dict: log_dict[key] = []
            log_dict[key].append(macro_metrics)

        # per-dataset metrics
        for task_name, tm in val_report['per_task'].items():
            task_metrics = tm['metrics']
            if isinstance(metric_fn, dict):
                for name, value in task_metrics.items():
                    key = f'val_{task_name}_{name}'
                    if key not in log_dict: log_dict[key] = []
                    log_dict[key].append(value)
            else:
                key = f'val_{task_name}_metric'
                if key not in log_dict: log_dict[key] = []
                log_dict[key].append(task_metrics)

        log_dict['epoch'].append(epoch)
        log_dict['train_loss'].append(train_loss)
        log_dict['val_loss'].append(val_loss)

        if isinstance(val_metric, dict) and monitor_metric:
            pbar.set_description(
                f"Epoch {epoch} : Train {train_loss:.4f} | Val {val_loss:.4f} | {monitor_metric} {val_metric[monitor_metric]:.4f}")
        else:
            pbar.set_description(
                f"Epoch {epoch} : Train {train_loss:.4f} | Val {val_loss:.4f} | Metric {val_metric:.4f}")

        # Early stopping based on the chosen metric
        if es.check_criteria(val_metric[monitor_metric], mMLP):
            print(f'[training] Early stop reached at epoch {es.best_step} with metric {es.best_value}')
            break

    # ------------------------------------------------------------------
    # Save best model + Logs 
    # ------------------------------------------------------------------

    best_embedder_dict = es.restore_best()
    mMLP.load_state_dict(best_embedder_dict)

    torch.save(mMLP.state_dict(), os.path.join(hparams['PhaseC_save_dir'], 'phaseC_mMLP.pt'))

    log_df = pandas.DataFrame(log_dict)
    log_df.to_csv(os.path.join(hparams['PhaseC_save_dir'], 'phaseC_training.csv'), index=False)

    return mMLP