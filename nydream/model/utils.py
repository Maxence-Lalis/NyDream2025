from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from skmultilearn.model_selection import iterative_train_test_split # type: ignore
import torch
import torch as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .gnn.graphnets_film_vector import FiLM


def is_film_param(name: str) -> bool:
    """Return **True** for parameters belonging to FiLM layers (frozen Phase‑A)."""
    return "film" in name.lower()


def freeze_params(model, predicate, freeze: bool = True):
    """Freeze / un‑freeze parameters based on *predicate(name)*."""
    for name, p in model.named_parameters():
        p.requires_grad = (not freeze) if predicate(name) else freeze


def build_optimizer(model, lr: float):
    """Adam over *trainable* params **plus** per‑dataset log‑variance weights."""
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    return torch.optim.Adam(params, lr=lr)

def init_film_identity(m):
    if isinstance(m, FiLM):
        last = m.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
            last.bias[: last.out_features // 2].fill_(10.0)

def build_optimizer_with_sched(model, lr: float, scheduler: str | None = None):
    params = [p for p in model.parameters() if p.requires_grad]
    optim  = torch.optim.Adam(params, lr=lr)

    if scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optim, mode="max", factor=0.5, patience=10, min_lr=1e-6)
    elif scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optim, T_0=10, T_mult=2, eta_min=lr/100)
    else:
        sched = None
    return optim, sched


# ----------------------------------------------------------------------
# Split Helpers
# ----------------------------------------------------------------------
def _iterative_split(labels: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative stratified split on *row* indices."""
    idx = np.arange(len(labels)).reshape(-1, 1)
    train, _, test, _ = iterative_train_test_split(idx, labels, test_size=test_size)
    return train.flatten(), test.flatten()

def stratified_distance_split_graph(
    labels: torch.Tensor,          # (N_rows, n_desc) – **WITHOUT** intensity
    cids:   torch.Tensor,          # (N_rows,)
    test_frac: float = 0.70,
    n_bins: int = 10,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Return (train_row_indices, test_row_indices)."""

    # ---------- group row-indices per CID -------------------------------
    cid2rows: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cids.tolist()):
        cid2rows.setdefault(cid, []).append(idx)

    dup_infos = []        # (cid, max_distance)
    for cid, rows in cid2rows.items():
        if len(rows) <= 1:
            continue                          # singleton → ignore for distance strat
        vecs = labels[rows]                  # (n_rep, n_desc)
        cos  = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0),
                                   dim=-1, eps=1e-8)
        dist = 1.0 - cos                     # (n_rep, n_rep)
        dup_infos.append((cid, dist.triu(diagonal=1).max().item()))

    if not dup_infos:                        # unlikely for DREAM
        raise ValueError("No duplicate CIDs found – stratified split not possible.")

    # ---------- quantile bins ------------------------------------------
    dists = torch.tensor([d for _, d in dup_infos])
    edges = torch.quantile(dists, torch.linspace(0, 1, n_bins + 1))
    bins  = torch.bucketize(dists, edges[1:-1], right=False).tolist()

    bin2cids: Dict[int, List[int]] = {}
    for (cid, _), b in zip(dup_infos, bins):
        bin2cids.setdefault(b, []).append(cid)

    g = torch.Generator().manual_seed(seed)
    test_dup_set: set[int] = set()
    for cids_in_bin in bin2cids.values():
        if not cids_in_bin:
            continue
        n_pick = max(1, int(round(test_frac * len(cids_in_bin))))
        perm   = torch.tensor(cids_in_bin)[torch.randperm(len(cids_in_bin), generator=g)].tolist()
        test_dup_set.update(perm[:n_pick])

    # ---------- final row allocation -----------------------------------
    train_idx, test_idx = [], []
    rng = torch.Generator().manual_seed(seed + 123)
    for cid, rows in cid2rows.items():
        if cid in test_dup_set:                  # exactly one replicate → test
            perm = torch.tensor(rows)[torch.randperm(len(rows), generator=rng)].tolist()
            test_idx.append(perm[0])
            train_idx.extend(perm[1:])
        else:                                    # all rows → train
            train_idx.extend(rows)

    # shuffle for reproducible batching
    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
    test_idx  = torch.tensor(test_idx )[torch.randperm(len(test_idx ), generator=g)].tolist()
    return train_idx, test_idx
# ──────────────────────────────────────────────────────────────


def stratified_distance_split_graph2(
    labels: torch.Tensor,        # (N_rows, n_desc)  – *no* intensity column
    cids:   torch.Tensor,        # (N_rows,)
    test_frac: float = 0.70,     # fraction of duplicate-CID *bins* sent to test
    n_bins:   int   = 10,
    seed:     int   = 0,
) -> Tuple[List[int], List[int]]:
    """
    Duplicate-aware split.  If **no** duplicate CIDs exist, we revert to a
    simple 20 % random CID split that still keeps all rows of a CID together.
    Returns two *row-index* lists: (train_idx, test_idx).
    """
    g = torch.Generator().manual_seed(seed)

    # ───────────────────────────────── duplicate analysis ───────────────
    cid2rows: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cids.tolist()):
        cid2rows.setdefault(cid, []).append(idx)

    dup_infos = []                                   # (cid, max_distance)
    for cid, rows in cid2rows.items():
        if len(rows) <= 1:
            continue
        vecs = labels[rows]                          # (n_rep, n_desc)
        cos  = F.cosine_similarity(
                   vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1, eps=1e-8)
        dist = 1.0 - cos
        dup_infos.append((cid, dist.triu(1).max().item()))

    # ───────────────────────────────── no duplicates  → simple split ────
    if not dup_infos:
        uniq_cids = torch.tensor(list(cid2rows.keys()))
        perm      = uniq_cids[torch.randperm(len(uniq_cids), generator=g)]
        n_test    = max(1, int(round(0.20 * len(uniq_cids))))   # 20 % rows
        test_set  = set(perm[:n_test].tolist())

        train_idx, test_idx = [], []
        for cid, rows in cid2rows.items():
            (test_idx if cid in test_set else train_idx).extend(rows)

        # row-order shuffle for reproducible batches
        train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
        test_idx  = torch.tensor(test_idx )[torch.randperm(len(test_idx ),  generator=g)].tolist()
        return train_idx, test_idx

    # ───────────────────────────────── duplicates present  ──────────────
    dists = torch.tensor([d for _, d in dup_infos])
    edges = torch.quantile(dists, torch.linspace(0, 1, n_bins + 1))
    bins  = torch.bucketize(dists, edges[1:-1], right=False).tolist()

    bin2cids: Dict[int, List[int]] = {}
    for (cid, _), b in zip(dup_infos, bins):
        bin2cids.setdefault(b, []).append(cid)

    test_dup_set: set[int] = set()
    for cids_in_bin in bin2cids.values():
        if not cids_in_bin:
            continue
        n_pick = max(1, int(round(test_frac * len(cids_in_bin))))
        perm   = torch.tensor(cids_in_bin)[torch.randperm(len(cids_in_bin), generator=g)]
        test_dup_set.update(perm[:n_pick].tolist())

    train_idx, test_idx = [], []
    rng = torch.Generator().manual_seed(seed + 123)
    for cid, rows in cid2rows.items():
        if cid in test_dup_set:                      # exactly one replicate → test
            perm = torch.tensor(rows)[torch.randperm(len(rows), generator=rng)].tolist()
            test_idx.append(perm[0]);  train_idx.extend(perm[1:])
        else:
            train_idx.extend(rows)

    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
    test_idx  = torch.tensor(test_idx )[torch.randperm(len(test_idx ),  generator=g)].tolist()
    return train_idx, test_idx



# ----------------------------------------------------------------------
# Metric Helpers
# ----------------------------------------------------------------------
def multilabel_pearson():
    """Mean Pearson‑r across label columns (ignores constant columns)."""

    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        targets = y_true.detach().cpu().numpy()
        rs = []
        for col in range(targets.shape[1]):
            if targets[:, col].std() == 0:
                continue  # undefined
            rs.append(np.corrcoef(probs[:, col], targets[:, col])[0, 1])
        return float(np.nanmean(rs)) if rs else np.nan

    return _metric

def cosine_loss(y_hat, y, eps=1e-8):
    cos = F.cosine_similarity(y_hat, y, dim=-1, eps=eps)   # (batch,)
    return (1.0 - cos).mean()

def base_losses(recon, y, mu, logvar):
    # import torch.nn as nn
    # bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # recon_l = bce_loss(recon, y)
    recon_l = F.mse_loss(recon, y, reduction="mean")
    kl_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    free_bits = 1
    return recon_l, torch.clamp(kl_l, min=free_bits)  # free‑bits min‑KL

def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=-1, eps=1e-8)


def pearson_corr(a, b):
    am, bm = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (am * bm).sum(-1) / (am.norm(dim=-1) * bm.norm(dim=-1) + 1e-8)


def _checkpoint_exists(save_dir: Path,
                       *names: str,
                       load_into: Tuple[torch.nn.Module, ...] | None = None
                      ) -> bool:
    """
    Returns True iff *all* files in `names` exist in save_dir.
    If `load_into` is given, it must have the same length as `names`;
    the matching state-dicts are loaded into those modules.
    """
    paths = [save_dir / n for n in names]
    ok = all(p.is_file() for p in paths)
    if ok and load_into is not None:
        for m, p in zip(load_into, paths):
            m.load_state_dict(torch.load(p, map_location="cpu"))
    return ok

# ----------------------------------------------------------------------
# Scaling Helpers
# ----------------------------------------------------------------------

def fit_minmax(t):
    t = torch.as_tensor(t, dtype=torch.float32)   # handles NumPy or Tensor
    return t.min(0).values, t.max(0).values

def transform(t: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    return (t - lo) / (hi - lo + 1e-8)

def inverse(t_scaled: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    return t_scaled * (hi - lo) + lo

def save_scaler(path: Path, lo: torch.Tensor, hi: torch.Tensor):
    np.savez(path, lo=lo.numpy(), hi=hi.numpy())

def load_scaler(path: Path):
    f = np.load(path)
    return torch.from_numpy(f["lo"]), torch.from_numpy(f["hi"])

def colwise_minmax(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (lo, hi) vectors with shape (D,) for a 2-D tensor (N, D)."""
    lo = t.min(dim=0).values
    hi = t.max(dim=0).values
    span = hi - lo
    span[span == 0] = 1.0          # avoid divide-by-zero for constant cols
    return lo, hi                  # each (D,)

def minmax_normalise(t: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (t - lo) / (hi - lo)

def minmax_inverse(t: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return t * (hi - lo) + lo