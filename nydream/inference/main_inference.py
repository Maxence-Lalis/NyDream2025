import torch
from torch_geometric.loader import DataLoader as pygdl
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import os
import pandas
import yaml

from model.gnn.graphnets_film_vector import GraphNets
from model.prediction_head.GLM import SimpleMLP
from model.cVAE.multicVAE import MultiHeadMLP
from dataloader.dataloader import DreamLoader
from model.gnn.predictor import EndToEndModule

from inference.make_inference_epoch import make_inference_epoch
from inference.inference_utils import save_inference

def main_inference(hparams):
    """
    """

    # ------------------------------------------------------------------
    # Load data 
    # ------------------------------------------------------------------

    dl = DreamLoader(full_data_path=hparams["Test_data_full_train"],
                smiles_column=hparams["Test_data_smiles"],
                label_column=hparams["Test_data_label"],
                id_column=hparams["Test_data_id"],
                conc_column=hparams['Test_conc_column'],
                save_dir=hparams["save_dir"],
                sep=hparams["Test_data_sep"],
                mode=hparams["Test_mode"],
                train_path=hparams['Test_data_train'],
                valid_path=hparams['Test_data_valid'],
                test_path=hparams['Test_data_test'],
                exclude_path=hparams['Test_data_exclude'],
                normalize_labels=hparams['Test_norm_method']
                )

    
    test_ds = dl.load_inference()
    test_ds.featurize('molecular_graphs_with_ec50_rep', init_globals=hparams['Test_init_globals'])
    print("[inference] test set size:", len(test_ds))

    inference_loader = pygdl(test_ds, batch_size=hparams['Test_batch_size'], shuffle=False)

    # ------------------------------------------------------------------
    # Make inference
    # ------------------------------------------------------------------

    with open(os.path.join(hparams['chkpt_dir'],"hparams.yaml"), "r", encoding="utf-8") as f:
        text = f.read()
    chkpt_params = yaml.safe_load(text)
    
    node_dim = test_ds.features[0].x.size(1) if hasattr(test_ds.features[0], "x") else 50
    edge_dim = test_ds.features[0].edge_attr.size(1) if hasattr(test_ds.features[0], "edge_attr") else 10
    global_dim = test_ds.features[0].u.size(1)
    
    gnn = GraphNets(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=chkpt_params['graphnet_hidden_dims'],
        depth=chkpt_params['graphnet_depth'],
        dropout=chkpt_params['graphnet_dropout'],
    ).to(hparams['Test_device'])

    pred = SimpleMLP(input_dim=global_dim, output_dim=chkpt_params['phaseA_data_label_size']).to(hparams['Test_device'])
    model = EndToEndModule(gnn, pred).to(hparams['Test_device'])

    if "A" in hparams['Phase_to_predict']:
        gnn_ckpt = os.path.join(hparams['chkpt_dir'],"phaseA","phaseA_gnn_embedder.pt")
        pred_ckpt = os.path.join(hparams['chkpt_dir'],"phaseA","phaseA_predictor.pt")

        model.gnn_embedder.load_state_dict(torch.load(gnn_ckpt, map_location=hparams['Test_device']))
        model.nn_predictor.load_state_dict(torch.load(pred_ckpt, map_location=hparams['Test_device']))
        model.eval()

        preds, embeds = make_inference_epoch(model, inference_loader, device=hparams['Test_device'])
        save_inference(preds, embeds, test_ds, hparams, phase_suffix="phaseA")

    if "B" in hparams['Phase_to_predict']:
        
        pred = SimpleMLP(input_dim=global_dim, output_dim=chkpt_params['PhaseB_data_label_size']).to(hparams['Test_device'])
        model = EndToEndModule(gnn, pred).to(hparams['Test_device'])

        gnn_ckpt = os.path.join(hparams['chkpt_dir'],"phaseB","phaseB_gnn_embedder.pt")
        pred_ckpt = os.path.join(hparams['chkpt_dir'],"phaseB","phaseB_predictor.pt")

        model.gnn_embedder.load_state_dict(torch.load(gnn_ckpt, map_location=hparams['Test_device']))
        model.nn_predictor.load_state_dict(torch.load(pred_ckpt, map_location=hparams['Test_device']))
        model.eval()

        preds, embeds = make_inference_epoch(model, inference_loader, device=hparams['Test_device'])
        save_inference(preds, embeds, test_ds, hparams, phase_suffix="phaseB")

    if "C" in hparams['Phase_to_predict']:
        gnn_ckpt = os.path.join(hparams['chkpt_dir'],"phaseB","phaseB_gnn_embedder.pt")
        pred_ckpt = os.path.join(hparams['chkpt_dir'],"phaseB","phaseB_predictor.pt")

        model.gnn_embedder.load_state_dict(torch.load(gnn_ckpt, map_location=hparams['Test_device']))
        model.nn_predictor.load_state_dict(torch.load(pred_ckpt, map_location=hparams['Test_device']))
        model.eval()
        
        embeddings = []
        for data in inference_loader:
            h = model.gnn_embedder(data.to(hparams['Test_device'])).detach().cpu()
            embeddings.append(h)

        X = torch.cat(embeddings)
        inference_loader_C = DataLoader(X, batch_size=hparams['Test_batch_size'])
        # inference_set = TensorDataset(X)
        # inference_loader_C = DataLoader(inference_set,   batch_size=hparams['Test_batch_size'])

        rata_dims = {t: cfg["out_dim"] for t, cfg in chkpt_params['PhaseC_panel'].items()}
        mMLP = MultiHeadMLP(
            embed_dim  = X.shape[1],
            rata_dims  = rata_dims,
            dropout    = chkpt_params['PhaseC_dropout'],
        ).to(hparams['Test_device'])

        mMLP_chkpt = os.path.join(hparams['chkpt_dir'],"phaseC","phaseC_mMLP.pt")
        mMLP.load_state_dict(torch.load(mMLP_chkpt, map_location=hparams['Test_device']))
        mMLP.eval()
        
        preds, embeds = make_inference_epoch(mMLP, inference_loader_C, device=hparams['Test_device'], taskname="DREAM")
        save_inference(preds, embeds, test_ds, hparams, phase_suffix="phaseC", embed=False)
