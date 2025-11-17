import torch
from torch.utils.data import Dataset

from typing import List, Iterable, Union, Optional, Callable

import numpy as np
import rdkit.Chem.AllChem as Chem
import torch_geometric as pyg

LATENT_SLICE = slice(202, 234)          # [202, 234) → 32 dims


def _keep_graph(g: pyg.data.Data) -> bool:
    """
    Return True when the graph's z-ec50 latent code is *not* all-zero.
    Assumes `g.u` (or `g.globals`) has shape [1, 202+32+1].
    """
    u = getattr(g, "u", getattr(g, "globals", None))
    if u is None or u.numel() == 0:
        return False                     # no globals at all → drop
    z = u[..., LATENT_SLICE]             # tensor of shape (1,32)
    return not torch.all(z == 0)         # keep only if some info is present


class GraphDataset(Dataset):
    def __init__(self,
                 x: List[pyg.data.Data],
                 y: Iterable[Union[int, float]]):
        # --- 1. build a mask -------------------------------------------------
        mask = [_keep_graph(g) for g in x]
        kept = sum(mask)
        if kept == 0:
            raise ValueError("Every graph has an empty z-ec50 – nothing to train on.")

        # --- 2. slice every parallel array with the mask ---------------------
        self.x   = [g  for g, m in zip(x,   mask) if m]
        self.y   = [v  for v, m in zip(y,   mask) if m]

        # --- 3. store dims from the *first* retained graph -------------------
        g0 = self.x[0]
        self.node_dim   = g0.x.shape[-1]
        self.edge_dim   = g0.edge_attr.shape[-1]
        self.global_dim = g0.u.shape[-1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return (self.x[idx],
                torch.tensor(self.y[idx], dtype=torch.float32))

    
class CIDGraphDataset(Dataset):
    def __init__(self,
                 x: List[pyg.data.Data],
                 y: Iterable[Union[int, float]],
                 cids: Iterable[int],
                 intensity: Iterable[float]):
        # --- 1. build a mask -------------------------------------------------
        mask = [_keep_graph(g) for g in x]
        kept = sum(mask)
        if kept == 0:
            raise ValueError("Every graph has an empty z-ec50 – nothing to train on.")
        print(kept, "This much molecules were kept")
        # --- 2. slice every parallel array with the mask ---------------------
        self.x   = [g  for g, m in zip(x,   mask) if m]
        self.y   = [v  for v, m in zip(y,   mask) if m]
        self.cids= [c  for c, m in zip(cids,mask) if m]
        self.intensity = [i for i, m in zip(intensity, mask) if m]

        # --- 3. store dims from the *first* retained graph -------------------
        g0 = self.x[0]
        self.node_dim   = g0.x.shape[-1]
        self.edge_dim   = g0.edge_attr.shape[-1]
        self.global_dim = g0.u.shape[-1]
        
        self.cid2pos = {c: p for p, c in enumerate(self.cids)}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int):
        return (self.x[idx],
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.cids[idx], dtype=torch.int32),
                torch.tensor(self.intensity[idx], dtype=torch.float32))
    
    def get_by_cid(self, cid):
        """Fetch sample by its CID instead of its position."""
        pos = self.cid2pos[cid]
        return self.__getitem__(pos)