import os
import pandas as pd 

def save_inference(preds, embeds, inference_loader, hparams, phase_suffix=None, embed=True):
    """
    Save predictions + embeddings (and each separately) to CSV.
    Optionally add a phase-specific suffix to filenames.
    """
    suffix = f"_{phase_suffix}" if phase_suffix is not None else ""

    # preds
    np_preds = preds.numpy()
    n_targets = np_preds.shape[1]
    pred_cols = [f"pred_{i}" for i in range(n_targets)]
    df_preds = pd.DataFrame(np_preds, columns=pred_cols)

    if embed :# embeds
        np_embeds = embeds.numpy()
        n_embeds = np_embeds.shape[1]
        embed_cols = [f"embed_{i}" for i in range(n_embeds)]
        df_embeds = pd.DataFrame(np_embeds, columns=embed_cols)

        # combined
        df = pd.concat([df_preds, df_embeds], axis=1)
        df['SMILES'] = list(inference_loader.smiles)
        df['CID'] = list(inference_loader.cids)
        out_path_combined = os.path.join(hparams['save_dir'], f"predictions{suffix}.csv")
        df.set_index("SMILES").to_csv(out_path_combined)
        
        # embeds only
        df_embeds['SMILES'] = list(inference_loader.smiles)
        df_embeds['CID'] = list(inference_loader.cids)
        out_path_embeds = os.path.join(hparams['save_dir'], f"predictions_embeds_only{suffix}.csv")
        df_embeds.set_index("SMILES").to_csv(out_path_embeds)

    # preds only
    df_preds['SMILES'] = list(inference_loader.smiles)
    df_preds['CID'] = list(inference_loader.cids)
    out_path_preds = os.path.join(hparams['save_dir'], f"predictions_preds_only{suffix}.csv")
    df_preds.set_index("SMILES").to_csv(out_path_preds)


    print(f"[inference] Wrote {len(df_preds)} predictions to {hparams['save_dir']}")