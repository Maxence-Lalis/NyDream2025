import ast
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split # type: ignore
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


def _safe_literal_eval(x):
    if not isinstance(x, str):
        return x
    s = x.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        # Handle space/newline-separated numerics inside or outside brackets
        inner = s[1:-1] if (s.startswith('[') and s.endswith(']')) else s
        arr = np.fromstring(inner.replace(',', ' '), sep=' ')
        if arr.size > 0:
            return arr.tolist()
        return s

def _ensure_2d_labels(y_col):
    """
    Convert a column of labels into a 2D numpy array:
    - If entries are lists/arrays -> stack
    - If entries are scalars -> one column
    - If entries are strings like '[0, 1, 0]' -> literal_eval then stack
    """
    processed = []
    for v in y_col:
        v = _safe_literal_eval(v)
        if isinstance(v, (list, tuple, np.ndarray)):
            processed.append(np.asarray(v, dtype=float).ravel())
        else:
            # scalar
            processed.append(np.asarray([float(v)], dtype=float))
    Y = np.vstack(processed)
    return Y

def _iterative_split(labels: np.ndarray, test_size: float, valid_size: float):
    """Iterative stratified split on *row* indices."""
    
    X = np.arange(len(labels)).reshape(-1, 1)  # dummy feature array for the splitter
    # ---- first split: trainval vs test ----
    test_fraction = test_size
    X_trainval, Y_trainval, X_test, Y_test = iterative_train_test_split(
        X, labels, test_fraction
    )

    # ---- second split: train vs valid ----
    valid_fraction = valid_size / (1.0 - test_size)
    X_train, Y_train, X_valid, Y_valid = iterative_train_test_split(
        X_trainval, Y_trainval, valid_fraction
    )

    train_idx = X_train.ravel()
    valid_idx = X_valid.ravel()
    test_idx = X_test.ravel()

    return train_idx, valid_idx, test_idx


def stratified_distance_split_graph(
    labels: torch.Tensor,          # (N_rows, n_desc) – no intensity column
    cids:   torch.Tensor,          # (N_rows,)
    test_frac: float = 0.70,       # fraction of duplicate-CID *bins* sent to test
    n_bins:   int   = 10,
    seed:     int   = 0,
) -> Tuple[List[int], List[int]]:
    """
    Duplicate-aware split (test gets exactly one replicate for selected CIDs).
    If NO duplicate CIDs exist, fall back to a simple 20% random CID split
    that keeps all rows of a CID together.

    Returns row-index lists: (train_idx, test_idx)
    """
    g = torch.Generator().manual_seed(seed)

    # Group rows by CID
    cid2rows: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cids.tolist()):
        cid2rows.setdefault(int(cid), []).append(idx)

    # For duplicate CIDs, compute max pairwise cosine distance
    dup_infos = []  # (cid, max_distance)
    for cid, rows in cid2rows.items():
        if len(rows) <= 1:
            continue
        vecs = labels[rows]  # (n_rep, n_desc)
        cos  = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1, eps=1e-8)
        dist = 1.0 - cos
        dup_infos.append((cid, dist.triu(diagonal=1).max().item()))

    # No duplicates → simple 20% CID split (keep CIDs intact)
    if not dup_infos:
        uniq_cids = torch.tensor(list(cid2rows.keys()))
        perm      = uniq_cids[torch.randperm(len(uniq_cids), generator=g)]
        n_test    = max(1, int(round(0.20 * len(uniq_cids))))
        test_set  = set(perm[:n_test].tolist())

        train_idx, test_idx = [], []
        for cid, rows in cid2rows.items():
            (test_idx if cid in test_set else train_idx).extend(rows)

        # Shuffle row order for reproducible batching
        train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
        test_idx  = torch.tensor(test_idx )[torch.randperm(len(test_idx ), generator=g)].tolist()
        return train_idx, test_idx

    # Duplicates present → quantile-bin the max distances and sample CIDs per bin
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

    # Allocate rows: for test CIDs → exactly one replicate to test, rest to train
    train_idx, test_idx = [], []
    rng = torch.Generator().manual_seed(seed + 123)
    for cid, rows in cid2rows.items():
        if cid in test_dup_set:
            perm = torch.tensor(rows)[torch.randperm(len(rows), generator=rng)].tolist()
            test_idx.append(perm[0])
            train_idx.extend(perm[1:])
        else:
            train_idx.extend(rows)

    # Shuffle for reproducible batching
    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx), generator=g)].tolist()
    test_idx  = torch.tensor(test_idx )[torch.randperm(len(test_idx ), generator=g)].tolist()
    return train_idx, test_idx
