from datetime import datetime
import yaml
import json
import os

def _read_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    suffix = ext.lower()
    
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    if suffix in {".yaml", ".yml"}:
        return dict(yaml.safe_load(text) or {})
    elif suffix == ".json":
        return json.loads(text or "{}")
    else:
        raise Exception(f"File format {suffix} not supported.")
    
def _update_and_save_config_train(hparams, date, run=None):
    
    base_dir = hparams['save_dir_init']
    model = hparams['model']
    mode = hparams['mode']
    tag = hparams['tag']

    final_dir = os.path.join(base_dir, model, f"{model}_{mode}_{tag}_{date}", f"run_{run}")
    os.makedirs(final_dir, exist_ok=True)
    hparams["save_dir"] = final_dir
    
    config_out = os.path.join(final_dir, "hparams.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)

    return hparams

def _update_and_save_config_predict(hparams, date, run=None):
    
    base_dir = hparams['save_dir_init']
    model = hparams['model']
    mode = hparams['mode']
    tag = hparams['tag']
    chkpt_dir = os.path.join(hparams['chkpt_dir_init'], f"run_{run}")
    
    final_dir = os.path.join(base_dir, model, f"{model}_{mode}_{tag}_{date}", f"run_{run}")
    os.makedirs(final_dir, exist_ok=True)
    
    hparams["save_dir"] = final_dir
    hparams["chkpt_dir"] = chkpt_dir

    config_out = os.path.join(final_dir, "hparams.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)

    return hparams

def _update_and_save_config_eval(hparams, date, run=None):
    
    base_dir = hparams['save_dir_init']
    model = hparams['model']
    mode = hparams['mode']
    tag = hparams['tag']
    chkpt_dir = os.path.join(hparams['chkpt_dir_init'], f"run_{run}")
    
    final_dir = os.path.join(base_dir, model, f"{model}_{mode}_{tag}_{date}", f"run_{run}")
    os.makedirs(final_dir, exist_ok=True)
    
    hparams["save_dir"] = final_dir
    hparams["chkpt_dir"] = chkpt_dir

    valid_path_dir = hparams['valid_path_dir']
    data_valid = os.path.join(valid_path_dir, f"run_{run}", "valid.csv")
    hparams['data_valid'] = data_valid

    config_out = os.path.join(final_dir, "hparams.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)

    return hparams

def _update_and_save_config_summary(hparams, date, run=None):
    
    hparams['date'] = date

    config_out = os.path.join(hparams['save_dir'], "hparams_summary.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)

    return hparams

def _update_and_save_config_train_nydream(hparams, date, run=None):
    
    base_dir = hparams['save_dir_init']
    model = hparams['model']
    mode = hparams['mode']
    tag = hparams['tag']

    final_dir = os.path.join(base_dir, model, f"{model}_{mode}_{tag}_{date}", f"run_{run}")
    if hparams['use_existing']:
        final_dir = hparams['use_existing']
    os.makedirs(final_dir, exist_ok=True)
    hparams["save_dir"] = final_dir
    
    phaseA_dir = os.path.join(final_dir, "phaseA")
    os.makedirs(phaseA_dir, exist_ok=True)
    phaseB_dir = os.path.join(final_dir, "phaseB")
    os.makedirs(phaseB_dir, exist_ok=True)
    phaseC_dir = os.path.join(final_dir, "phaseC")
    os.makedirs(phaseC_dir, exist_ok=True)

    hparams["PhaseA_save_dir"] = phaseA_dir
    hparams["PhaseB_save_dir"] = phaseB_dir
    hparams["PhaseC_save_dir"] = phaseC_dir
    
    for name, task in hparams["PhaseC_panel"].items():
        if not str(task.get("save_dir") or "").strip():
            task["save_dir"] = os.path.join(phaseC_dir,name)

    config_out = os.path.join(final_dir, "hparams.yaml")
    with open(config_out, "w") as f:
        yaml.safe_dump(hparams, f, sort_keys=False)

    return hparams
