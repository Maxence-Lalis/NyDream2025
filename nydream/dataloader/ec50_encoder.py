import torch, pickle, re
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import numpy

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



def parse_or(seq_id: str) -> int:
    import re
    m = re.match(r"s_(\d+)", seq_id)
    if not m:
        raise ValueError(f"cannot parse OR id from {seq_id}")
    return int(m.group(1))




# ═════════════════════════════════════ dataset ──
class ORSetDataset(torch.utils.data.Dataset):
    """Each item  →  (ids, p_norm, r)

    * **ids**   : tensor [N] receptor IDs shifted by +1 (0 = padding)
    * **p_norm**: tensor [N] normalised pEC50 values
    * **r**     : tensor [N] raw Rmax values (0‑1)
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
        self.or_emb = nn.Embedding(n_or, latent)         # 0..n_or‑1 (no pad)
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

_OR_PATTERN = re.compile(r"s_(\d+)")   # parses “s_97” → 97

class EC50EncoderV2:
    """
    Usage
    -----
    enc = EC50EncoderV2("best_set_autoenc2.pt", device="cuda")
    z   = enc.encode(ec50_dataframe, cid=123456)
    """
    def __init__(self, ckpt_path: str, device: str = "cpu", n_or: int = 385):
        self.device = device
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt["state"]

        # ── 0 · strip “module.” if DataParallel was used ───────────────────
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}

        # ── 1 · infer sizes from weights in the checkpoint ─────────────────
        emb_w = state["enc.or_emb.weight"]          # shape = [n_or+1, id_dim]
        id_dim = emb_w.size(1)
        latent = state["dec.or_emb.weight"].size(1)

        # ── 2 · rebuild network skeleton and load weights ─────────────────
        self.model = ORAutoEncoder(n_or=n_or,
                                   id_dim=id_dim,
                                   latent=latent).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

        # ── 3 · store normalisation constants ─────────────────────────────
        self.mean_p = ckpt.get("mean_p", 0.0)
        self.std_p  = ckpt.get("std_p",  1.0)

    # ── util: build model inputs for ONE molecule ─────────────────────────
    def _build_inputs(self, df_mol, rmax_thresh=0.75):
        """
        Parameters
        ----------
        df_mol : DataFrame with index “_seq_id”, columns “ec50”, “top”, …
        Returns
        -------
        ids          : [N]   LongTensor (+1 shift, 0 = pad)
        p_norm, rmax : [N]   FloatTensors
        pad_mask     : [N]   BoolTensor (False = real, True = pad)
        or  (None, …) if no active receptor above threshold.
        """
        actives = df_mol[df_mol["top"] >= rmax_thresh]
        if actives.empty:
            return None, None, None, None

        # ---- receptor IDs (+1 shift) ----
        ids = torch.tensor(
            [int(_OR_PATTERN.match(idx).group(1)) + 1 for idx in actives.index],
            dtype=torch.long, device=self.device)

        # ---- pEC50 normalised ----
        ec50 = actives["ec50"].clip(lower=1e-12).to_numpy()
        p_norm = torch.tensor(
            (-np.log10(ec50) - self.mean_p) / self.std_p,
            dtype=torch.float32, device=self.device)

        # ---- Rmax & mask ----
        rmax = torch.tensor(actives["top"].to_numpy(),
                            dtype=torch.float32, device=self.device)
        pad_mask = torch.zeros_like(ids, dtype=torch.bool)   # no padding
        return ids, p_norm, rmax, pad_mask

    # ── public API ────────────────────────────────────────────────────────
    def encode(self, ec50_df, cid, rmax_thresh=0.75):
        """
        ec50_df : big DataFrame with a “CID” column
        cid      : compound identifier to embed
        Returns  : 1-D latent tensor on self.device  (zeros if no actives)
        """
        df_mol = ec50_df[ec50_df["CID"] == cid].reset_index().set_index("_seq_id")
        ids, p_norm, rmax, pad_mask = self._build_inputs(df_mol, rmax_thresh)

        if ids is None:                                  # no active ORs
            lat_dim = self.model.enc.st.dec[-1].out_features
            return torch.zeros(lat_dim, device=self.device)

        # Model expects batched [B, L] tensors
        z = self.model.enc(ids.unsqueeze(0),
                           p_norm.unsqueeze(0),
                           rmax.unsqueeze(0),
                           pad_mask.unsqueeze(0))        # ⇒ [1, latent]
        return z.squeeze(0)                              # [latent]