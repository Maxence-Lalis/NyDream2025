import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
from master_utils import _read_config, _update_and_save_config_train, _update_and_save_config_predict, _update_and_save_config_eval, _update_and_save_config_summary, _update_and_save_config_train_nydream

def main(args):
    
    print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
    
    hparams = _read_config(args.config)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")


    # ------------------------------------------------------------------
    # Train NyDream
    # ------------------------------------------------------------------

    if (hparams['mode'] == "train") & (hparams['model'] == "nydream"):
        from training.main_train_eval import main_train_eval
        for i in hparams['runs']:
            print(f"[training] run {i} ...")
            new_hparams = _update_and_save_config_train_nydream(hparams, date_str, run=i)
            main_train_eval(new_hparams)

    # ------------------------------------------------------------------
    # Predict NyDream
    # ------------------------------------------------------------------

    elif (hparams['mode'] == "inference") & (hparams['model'] == "nydream"):
        from inference.main_inference import main_inference
        for i in hparams['runs']:
            print(f"[inference] run {i} ...")
            new_hparams = _update_and_save_config_predict(hparams, date_str, run=i)
            main_inference(new_hparams)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    main(args)