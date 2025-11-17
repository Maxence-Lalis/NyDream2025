# prediction_head/simple.py
"""
A *tiny* helper module that hard-codes everything for **multilabel**
classification with per–label BCE-with-logits loss.

No TaskType enum, no TaskSpec, no per-task dicts.
"""

from __future__ import annotations
from pathlib import Path
import sys

from sklearn.metrics import f1_score
import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------- #
# 1.  Fixed “settings”                                                   #
# --------------------------------------------------------------------- #

ACTIVATION   : nn.Module  = nn.Identity()          # logits straight out
LOSS_FN      : nn.Module  = nn.BCEWithLogitsLoss() # one loss for the whole tensor
THRESHOLD               = 0.5                      # for metrics (sigmoid > 0.5)

# --------------------------------------------------------------------- #
# 2.  Tiny metric helpers                                                #
# --------------------------------------------------------------------- #

def multilabel_macro_f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Macro F1 after applying sigmoid + 0.5 threshold.
    """
    y_pred_bin = (torch.sigmoid(y_pred) > THRESHOLD).cpu().numpy()
    return f1_score(y_true.cpu().numpy(), y_pred_bin, average="macro")

# You can add more metrics here if you need them
METRICS = {
    "macro_f1": multilabel_macro_f1,
}

# --------------------------------------------------------------------- #
# 3.  Utility wrappers used elsewhere                                    #
# --------------------------------------------------------------------- #

def get_activation():               # keeps the old call-style alive
    return ACTIVATION.__class__     # return *class*, not instance, like before

def get_loss_fn():                  # idem
    return LOSS_FN

def get_metrics():
    return METRICS
