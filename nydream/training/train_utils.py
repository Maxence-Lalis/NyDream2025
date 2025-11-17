from typing import Tuple
from pathlib import Path
import torch
from model.gnn.graphnets_film_vector import FiLM
import os

def _checkpoint_exists(save_dir,
                       *names: str,
                       load_into: Tuple[torch.nn.Module, ...] | None = None
                      ) -> bool:
    """
    Returns True iff *all* files in `names` exist in save_dir.
    If `load_into` is given, it must have the same length as `names`;
    the matching state-dicts are loaded into those modules.
    """
    paths = [os.path.join(save_dir, n) for n in names]
    ok = all(Path(p).is_file() for p in paths)    
    if ok and load_into is not None:
        for m, p in zip(load_into, paths):
            m.load_state_dict(torch.load(p, map_location="cpu"))
    return ok

def init_film_identity(m):
    if isinstance(m, FiLM):
        last = m.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[: last.out_features // 2].fill_(10.0)

def init_film_high_grad(m):
    if isinstance(m, FiLM):
        last = m.net[-1] 
        out = last.out_features
        half = out // 2
        with torch.no_grad():
            last.weight.normal_(mean=0.0, std=0.02) 
            last.bias.zero_()
            last.bias[:half].fill_(0.0)
            last.bias[half:].fill_(0.0)

def is_film_param(name: str) -> bool:
    """Return **True** for parameters belonging to FiLM layers (frozen Phase-A)."""
    return "film" in name.lower()

def freeze_params(model, predicate, freeze: bool = True):
    """Freeze / un-freeze parameters based on *predicate(name)*."""
    for name, p in model.named_parameters():
        p.requires_grad = (not freeze) if predicate(name) else freeze

def _add_tiny_noise_to_film_scales(module, std=1e-3):
    """
    Assumes FiLM's last Linear outputs [gamma || beta].
    Adds tiny noise to gamma bias to break symmetry.
    Works even if FiLM class isn't directly imported by checking class name.
    """
    if module.__class__.__name__.lower() == "film":
        last = module.net[-1]
        out = last.out_features
        half = out // 2
        with torch.no_grad():
            # gamma slice
            last.bias[:half].add_(std * torch.randn_like(last.bias[:half]))


def _build_param_groups(model, base_lr, head_lr, film_lr, weight_decay):
    film_params, base_params, head_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("nn_predictor") or "nn_predictor" in name:
            head_params.append(p)
        elif is_film_param(name):
            film_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        {"params": film_params, "lr": film_lr, "weight_decay": weight_decay},
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
    return param_groups


def _set_requires_grad(model, *, film=True, head=True, base=False):
    """Freeze/unfreeze partitions: FiLM, head (predictor), base (embedder)."""
    for name, p in model.named_parameters():
        if name.startswith("nn_predictor") or "nn_predictor" in name:
            p.requires_grad = head
        elif is_film_param(name):
            p.requires_grad = film
        else:
            p.requires_grad = base
