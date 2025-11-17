#!/usr/bin/env python3
"""train_or_autoencoder_fixed.py

Refactored version of the OR-set auto-encoder that addresses the three
critical issues highlighted in the review:

1. **Trainable receptor ID embedding** - the embedding is now part of the
   neural network (inside `ActiveORSetEncoder`) instead of the `Dataset`, so
   gradients flow and the vectors learn.
2. **Class-imbalance aware classifier** - `BCEWithLogitsLoss` is given a
   per-receptor `pos_weight` computed from the training data, preventing the
   model from collapsing to the *all-inactive* solution.
3. **Deterministic split & seeds** - reproducibility seeds are set for Python,
   NumPy and PyTorch, and the train/val split is driven by a seeded RNG so that
   metrics are comparable across runs. (Full scaffold-aware splitting is left
   as future work because SMILES are not available in the current Parquet.)

Usage (identical CLI save for two new flags):

```bash
python train_or_autoencoder_fixed.py \
       --parquet data/all_or_responses.parquet \
       --epochs 60 --latent 64 --cuda
```
"""
import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds  # streaming Parquet reader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# ─────────────────────────────────────────────────────────── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument("--parquet", required=True,
                    help="Parquet file with 2-level index [_mol_id,_seq_id]")
parser.add_argument("--rmax", type=float, default=0.20,
                    help="threshold on `top` to call a pair active")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--latent", type=int, default=32)
parser.add_argument("--id_dim", type=int, default=64,
                    help="size of the trainable receptor ID embedding vector")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()

# ───────────────────────────────────────────────────── reproducibility ──
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

DEVICE = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
print(DEVICE)
# ═══════════════════════════════════ SetTransformer layers (unchanged) ══
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        ds = self.dim_V // self.num_heads
        Q_, K_, V_ = (torch.cat(x.split(ds, 2), 0) for x in (Q, K, V))
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if hasattr(self, "ln0"):  # layer norm optional
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        if hasattr(self, "ln1"):
            O = self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        # X: [B, N, D_in] → out: [B, k, D_out]
        return self.dec(self.enc(X))

# ═════════════════════════════════════ helper functions ══
def compute_metrics(logit_act, act_true,
                    p_hat, p_true,
                    r_hat, r_true):
    probs = torch.sigmoid(logit_act).flatten().cpu().numpy()
    y_act = act_true.flatten().cpu().numpy()
    aupr = average_precision_score(y_act, probs)
    auc = roc_auc_score(y_act, probs)
    mask = act_true.bool()
    mae_p = torch.abs(p_hat[mask] - p_true[mask]).mean().item() if mask.any() else math.nan
    rmse_p = torch.sqrt(((p_hat[mask] - p_true[mask]) ** 2).mean()).item() if mask.any() else math.nan
    mae_r = torch.abs(r_hat[mask] - r_true[mask]).mean().item() if mask.any() else math.nan
    rmse_r = torch.sqrt(((r_hat[mask] - r_true[mask]) ** 2).mean()).item() if mask.any() else math.nan
    return dict(AUPRC=aupr, AUROC=auc,
                MAE_pEC50=mae_p, RMSE_pEC50=rmse_p,
                MAE_Rmax=mae_r, RMSE_Rmax=rmse_r)


def parse_or(seq_id: str) -> int:
    import re
    m = re.match(r"s_(\d+)", seq_id)
    if not m:
        raise ValueError(f"cannot parse OR id from {seq_id}")
    return int(m.group(1))


def parquet_to_dict(path: Path, rmax_thresh: float = 0.20,
                    log_ec50_floor: float = -12.0):
    """Stream Parquet, keep rows with top ≥ rmax_thresh, return
    {mol_id: [(or_id, pEC50, Rmax), …]}."""
    out = defaultdict(list)
    for batch in ds.dataset(path).to_batches(batch_size=250_000):
        df = batch.to_pandas()
        df["pEC50"] = -df["ec50"]
        df["or_id"] = df.index.get_level_values("_seq_id").map(parse_or)
        active = df[df["top"] >= rmax_thresh]
        for mol_id, blk in active.groupby(level=0):
            recs = blk[["or_id", "pEC50", "top"]].itertuples(index=False, name=None)
            out[mol_id].extend(recs)
    return dict(out)

# ═════════════════════════════════════ dataset ──
class ORSetDataset(torch.utils.data.Dataset):
    """Each item  →  (ids, p_norm, r)

    * **ids**   : tensor [N] receptor IDs shifted by +1 (0 = padding)
    * **p_norm**: tensor [N] normalised pEC50 values
    * **r**     : tensor [N] raw Rmax values (0-1)
    """
    def __init__(self, ec50_dict: dict, n_or: int, id_dim: int = 64):
        super().__init__()
        self.rows = list(ec50_dict.items())
        self.n_or = n_or
        self.id_dim = id_dim

        # statistics for pEC50 normalisation
        all_p = [p for _, recs in self.rows for (_, p, _) in recs]
        self.mean_p = float(torch.tensor(all_p).mean())
        self.std_p = float(torch.tensor(all_p).std())

        # positive counts per receptor (for pos_weight)
        self.pos_counts = np.zeros(n_or, dtype=np.int64)
        for _, recs in self.rows:
            for or_id, _, _ in recs:
                self.pos_counts[or_id] += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        mol_id, recs = self.rows[idx]
        or_ids = torch.tensor([oid + 1 for oid, _, _ in recs], dtype=torch.long)  # +1 shift, 0=pad
        p_norm = torch.tensor([(p - self.mean_p) / self.std_p for _, p, _ in recs], dtype=torch.float32)
        r_vals = torch.tensor([r for _, _, r in recs], dtype=torch.float32)
        return or_ids, p_norm, r_vals, {"mol_id": mol_id}


def collate_batch(batch):
    ids_lst, p_lst, r_lst, metas = zip(*batch)
    Ls = [x.size(0) for x in ids_lst]
    Lm = max(Ls)

    B = len(batch)
    ids_pad = torch.zeros(B, Lm, dtype=torch.long)        # 0 = padding_idx
    p_pad = torch.zeros(B, Lm, dtype=torch.float32)
    r_pad = torch.zeros(B, Lm, dtype=torch.float32)
    mask = torch.ones(B, Lm, dtype=torch.bool)            # True → padding

    for i, (ids, p, r) in enumerate(zip(ids_lst, p_lst, r_lst)):
        L = ids.size(0)
        ids_pad[i, :L] = ids
        p_pad[i, :L] = p
        r_pad[i, :L] = r
        mask[i, :L] = False
    return ids_pad, p_pad, r_pad, mask, metas

# ═════════════════════════════════════ model ──
class ActiveORSetEncoder(nn.Module):
    def __init__(self, n_or: int, id_dim: int, latent: int,
                 num_heads: int = 4, num_inds: int = 32, ln: bool = True):
        super().__init__()
        self.or_emb = nn.Embedding(n_or + 1, id_dim, padding_idx=0)  # +1 for pad token 0
        token_dim = id_dim + 2  # p_norm and Rmax scalars
        self.st = SetTransformer(token_dim, num_outputs=1, dim_output=latent,
                                 num_inds=num_inds, dim_hidden=4 * token_dim,
                                 num_heads=num_heads, ln=ln)

    def forward(self, ids, p_norm, r, pad_mask):
        """ids/p_norm/r: [B, L] ; pad_mask: [B, L] (True=pad).
        Returns z: [B, latent]."""
        id_vec = self.or_emb(ids)                        # [B, L, id_dim]
        tok = torch.cat([id_vec, p_norm.unsqueeze(-1), r.unsqueeze(-1)], dim=-1)
        tok = tok.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z = self.st(tok)                                 # [B, 1, latent]
        return z.squeeze(1)

class ORDecoder(nn.Module):
    def __init__(self, latent: int, n_or: int, hidden: int = 128):
        super().__init__()
        self.or_emb = nn.Embedding(n_or, latent)         # 0..n_or-1 (no pad)
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent, hidden), nn.SELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 3)
        )

    def forward(self, z):
        B, latent = z.size()
        ids = torch.arange(self.or_emb.num_embeddings, device=z.device)
        e = self.or_emb(ids).unsqueeze(0).expand(B, -1, -1)  # [B, N_or, latent]
        z_exp = z.unsqueeze(1).expand_as(e)
        out = self.mlp(torch.cat([z_exp, e], -1))
        return out[..., 0], out[..., 1], out[..., 2]  # logits, p_hat_norm, r_hat

class ORAutoEncoder(nn.Module):
    def __init__(self, n_or: int, id_dim: int, latent: int):
        super().__init__()
        self.enc = ActiveORSetEncoder(n_or, id_dim, latent)
        self.dec = ORDecoder(latent, n_or)

    def forward(self, ids, p_norm, r, mask, reconstruct: bool = False):
        z = self.enc(ids, p_norm, r, mask)
        return self.dec(z) if reconstruct else z

# ═════════════════════════════════════ training / evaluation ──

def train_one_epoch(model, loader, optimizer, pos_weight, λ_p=1.0, λ_r=1.0):
    model.train()
    tot_loss, n_samples = 0.0, 0
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    for ids, p_norm, r, pad_mask, _ in loader:
        ids, p_norm, r, pad_mask = (x.to(DEVICE) for x in (ids, p_norm, r, pad_mask))
        logits, p_hat_n, r_hat = model(ids, p_norm, r, pad_mask, reconstruct=True)

        B, N = logits.shape
        act_gt = torch.zeros_like(logits)
        p_gt = torch.zeros_like(p_hat_n)
        r_gt = torch.zeros_like(r_hat)

        # scatter ground truth without Python loops
        valid = ~pad_mask
        batch_idx, elem_idx = torch.where(valid)
        or_ids = ids[batch_idx, elem_idx] - 1             # shift back to 0..N-1
        act_gt[batch_idx, or_ids] = 1.0
        p_gt[batch_idx, or_ids] = p_norm[batch_idx, elem_idx]
        r_gt[batch_idx, or_ids] = r[batch_idx, elem_idx]

        cls_loss = bce(logits, act_gt)
        mse_p = mse(p_hat_n[act_gt.bool()], p_gt[act_gt.bool()]) if act_gt.any() else 0.0
        mse_r = mse(r_hat[act_gt.bool()], r_gt[act_gt.bool()]) if act_gt.any() else 0.0
        loss = cls_loss + λ_p * mse_p + λ_r * mse_r

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item() * B
        n_samples += B
    return tot_loss / n_samples

@torch.no_grad()

def evaluate(model, loader, mean_p, std_p):
    model.eval()
    agg, n_batches = defaultdict(float), 0
    for ids, p_norm, r, pad_mask, _ in loader:
        ids, p_norm, r, pad_mask = (x.to(DEVICE) for x in (ids, p_norm, r, pad_mask))
        logits, p_hat_n, r_hat = model(ids, p_norm, r, pad_mask, reconstruct=True)

        B, N = logits.shape
        act_gt = torch.zeros_like(logits)
        p_gt = torch.zeros_like(p_hat_n)
        r_gt = torch.zeros_like(r_hat)

        valid = ~pad_mask
        batch_idx, elem_idx = torch.where(valid)
        or_ids = ids[batch_idx, elem_idx] - 1
        act_gt[batch_idx, or_ids] = 1.0
        p_gt[batch_idx, or_ids] = p_norm[batch_idx, elem_idx]
        r_gt[batch_idx, or_ids] = r[batch_idx, elem_idx]

        p_hat = p_hat_n * std_p + mean_p
        p_gt_abs = p_gt * std_p + mean_p

        metrics = compute_metrics(logits.cpu(), act_gt.cpu(),
                                  p_hat.cpu(), p_gt_abs.cpu(),
                                  r_hat.cpu(), r_gt.cpu())
        for k, v in metrics.items():
            agg[k] += v
        n_batches += 1
    return {k: v / n_batches for k, v in agg.items()}

# ═════════════════════════════════════ MAIN ──
print("Streaming Parquet and building active-pair dictionary …")
ec50_dict = parquet_to_dict(Path(args.parquet), rmax_thresh=args.rmax)
print(f"✔ {len(ec50_dict):,} molecules with ≥1 active receptor")

N_OR = 385  # update here if your dataset changes

dataset = ORSetDataset(ec50_dict, N_OR, id_dim=args.id_dim)

# ── class-imbalance aware pos_weight ──
neg_counts = len(dataset) - dataset.pos_counts
pos_counts = dataset.pos_counts.copy()
pos_counts[pos_counts == 0] = 1  # avoid division by zero
pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32, device=DEVICE)

# ── deterministic 80/20 split ──
indices = np.arange(len(dataset))
np.random.shuffle(indices)
cut = int(0.8 * len(indices))
train_idx, val_idx = indices[:cut], indices[cut:]

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        collate_fn=collate_batch, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        collate_fn=collate_batch)

model = ORAutoEncoder(N_OR, args.id_dim, latent=args.latent).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

best = 0.0
for ep in range(1, args.epochs + 1):
    tr_loss = train_one_epoch(model, train_loader, optimizer, pos_weight)
    val_metrics = evaluate(model, val_loader, dataset.mean_p, dataset.std_p)
    print(f"E{ep:02d}  loss={tr_loss:.4f}  AUPRC={val_metrics['AUPRC']:.3f} AUROC={val_metrics['AUROC']:.3f} MAE_pEC50={val_metrics['MAE_pEC50']:.3f}")

    if val_metrics["AUPRC"] > best:
        best = val_metrics["AUPRC"]
        torch.save({"state": model.state_dict(),
                    "mean_p": dataset.mean_p,
                    "std_p": dataset.std_p},
                   "best_set_autoenc3.pt")

print("Best validation AUPRC:", best)
