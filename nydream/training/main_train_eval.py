import torch
from torch_geometric.loader import DataLoader as pygdl
from torch import nn

import pandas
import os
import tqdm
import numpy as np

from training.train_phaseA import phaseA_load_and_run
from training.train_phaseB import phaseB_load_and_run
from training.train_phaseC import phaseC_load_and_run
 
def main_train_eval(hparams):
    """
    """

    # Phase A
    model = phaseA_load_and_run(hparams)
    # Phase B
    model, loaders = phaseB_load_and_run(model, hparams)
    # Phase C
    phaseC_load_and_run(model,loaders, hparams)
