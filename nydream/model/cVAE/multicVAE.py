# Model/cVAE/multi_head_cvae.py
from typing import Dict, Sequence
import torch, torch.nn.functional as F
from torch import nn
from typing import Dict, Sequence
import torch
from torch import nn, Tensor


class MultiHeadMLP(nn.Module):
    """Multi‑head feed‑forward regressor mapping a GNN (or any) embedding → a RATA panel.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embedding ``x``.
    rata_dims : Dict[str, int]
        Mapping ``{task_name: output_dim}``, e.g. ``{"POM": 51, "DREAM": 55, "KELLER": 48}``.
        A separate linear head is created for every entry.
    hidden_dims : Sequence[int], default ``(512, 256)``
        Sizes of the shared hidden layers.
    dropout : float, default ``0.1``
        Dropout probability applied after every hidden layer.
    """

    def __init__(
        self,
        embed_dim: int,
        rata_dims: Dict[str, int],
        hidden_dims: Sequence[int] = (512,256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Save meta information ────────────────────────────
        self.rata_dims = rata_dims

        # ── Optional layer‑norm on the raw embedding ────────
        self.emb_norm = nn.LayerNorm(embed_dim)

        # ── Shared trunk (identical for every task) ─────────
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # ── One lightweight head per dataset / task ─────────
        self.heads = nn.ModuleDict({
            task: nn.Linear(in_dim, out_dim) for task, out_dim in rata_dims.items()
        })

    # ────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────
    def forward(self, x: Tensor, *, task: str) -> Tensor:
        """Predict a RATA panel given an embedding.

        Parameters
        ----------
        x : Tensor, shape (B, embed_dim)
            Input embedding.
        task : str
            Name of the RATA panel (must be a key of ``self.rata_dims``).

        Returns
        -------
        Tensor, shape (B, rata_dims[task])
            Predicted RATA intensities for the requested panel.
        """
        x = self.emb_norm(x)
        h = self.trunk(x)
        return self.heads[task](h)


from typing import Dict, Sequence
import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """2‑layer residual MLP block with LayerNorm in every sub‑layer."""

    def __init__(self, dim: int, p_drop: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.SiLU(),
            #nn.Dropout(p_drop),
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim),
            nn.SiLU(),
        )
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: Tensor) -> Tensor:  # shape (B, dim)
        return x + self.dropout(self.block(x))


class MultiHeadMLP2(nn.Module):
    """Shared MLP trunk + FiLM‑modulated head for multi‑panel RATA prediction.

    **Key extras compared with the vanilla version**
    * LayerNorm after *every* linear (incl. emb_norm on raw input).
    * One *ResidualBlock* to enlarge effective depth without hurting optimisation.
    * FiLM modulation (`γ`, `β`) learnt *per task* to adapt the shared representation
      before a *single* output projection.  This removes three separate heads while
      still giving each panel its own affine sub‑space.
    """

    def __init__(
        self,
        embed_dim: int,
        rata_dims: Dict[str, int],
        hidden_dims: Sequence[int] = (512,256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.rata_dims = rata_dims
        self.emb_norm = nn.LayerNorm(embed_dim)

        h1, h2 = hidden_dims
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, h1, bias=False),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(h1, p_drop=dropout),
            nn.Linear(h1, h2, bias=False),
            nn.LayerNorm(h2),
            nn.SiLU(),
        )

        # ── FiLM parameters per task ───────────────────────────────
        self.gamma = nn.ParameterDict({
            t: nn.Parameter(torch.ones(h2)) for t in rata_dims
        })
        self.beta = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(h2)) for t in rata_dims
        })
        # After registering γ, β
        for g in self.gamma.values(): nn.init.ones_(g)
        for b in self.beta.values():  nn.init.zeros_(b)
        # One *shared* projection. We cut on the fly for each task.
        self.out_proj = nn.Linear(h2, max(rata_dims.values()))

    # ────────────────────────────────────────────────────────────
    def forward(self, x: Tensor, *, task: str) -> Tensor:
        """Predict a RATA panel for the requested task.

        Parameters
        ----------
        x : Tensor, shape (B, embed_dim)
            Input molecule embedding.
        task : str
            Task/panel name (must exist in ``self.rata_dims``).
        """
        x = self.emb_norm(x)
        h = self.trunk(x)                      # (B, hidden_dim)
        h = h * self.gamma[task] + self.beta[task]
        out = self.out_proj(h)                 # (B, max_dim)
        return out[..., : self.rata_dims[task]]



class MultiHeadCVAE(nn.Module):
    """
    Conditional VAE with one decoder per Rate-All-That-Apply (RATA) panel.
    ─────────────────────────────────────────────────────────────────────
    Args
    ----
    embed_dim   : dimension of the GNN-concentration embedding (x)
    latent_dim  : size of z
    rata_dims   : dict {task_name → output_dim of that RATA panel}
                  e.g. {"DREAM55": 55, "RATA48": 48, "RATA56": 56}
    hidden_dims : (h₁, h₂) sizes used for encoder / decoder MLPs
    dropout     : p(drop) everywhere

    Notes
    -----
    • Encoder still sees (x, y)  ➜  q(z|x,y) but because y has
      variable length we *right-pad* it to `cond_dim = max(rata_dims)`.
    • Each task owns its own decoder; the latent space is shared.
    • Forward needs a `task` flag so we know which decoder to call.
    """
    def __init__(
        self,
        embed_dim:  int,
        latent_dim: int,
        rata_dims:  Dict[str, int],
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rata_dims = rata_dims
        self.cond_dim  = 51    # length after padding

        self.emb_norm = nn.LayerNorm(embed_dim)

        self.pom_head = nn.Linear(rata_dims['POM'], 51)
        self.dream_head = nn.Linear(rata_dims['DREAM'], 51)
        self.keller_head = nn.Linear(rata_dims['KELLER'], 51)

        # self.keller_head = nn.Sequential(nn.Linear(rata_dims['KELLER'], 128),nn.ReLU(),nn.Linear(128, 48))
        # self.pom_head = nn.Sequential(nn.Linear(rata_dims['POM'], 128),nn.ReLU(),nn.Linear(128, 48))
        # self.dream_head = nn.Sequential(nn.Linear(rata_dims['DREAM'], 128),nn.ReLU(),nn.Linear(128, 48))


        # ── Encoder ───────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim + self.cond_dim, hidden_dims[0]),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        self.mu     = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[1], latent_dim)

        # ── One decoder per task ─────────────────────────────────
        self.decoders = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(embed_dim + latent_dim, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], out_dim),
                # nn.Sigmoid(),                     # keep your original range
                nn.Identity(),
            )
            for task, out_dim in rata_dims.items()
        })


    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _pad_y(self, y: torch.Tensor, task: str) -> torch.Tensor:
        if y.size(-1) == self.cond_dim:        # already full length
            return y
        out_dim = self.rata_dims[task]
        pad = (0, self.cond_dim - out_dim)
        return F.pad(y, pad, value=0.0)

    # ────────────────────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────────────────────
    def forward(self, x, y=None, *, task: str):
        """
        During training supply both (x, y) and task name.
        During inference just give (x) and task.
        """
        x = self.emb_norm(x)

        if self.training:
            if task == "DREAM":
                out_dim = self.rata_dims[task]
                y = y[:, :out_dim]
                y_pad = self.dream_head(y)
            elif task == "POM":
                out_dim = self.rata_dims[task]
                y = y[:, :out_dim]
                y_pad = self.pom_head(y)
            elif task == "KELLER":
                out_dim = self.rata_dims[task]
                y = y[:, :out_dim]
                y_pad = self.keller_head(y)
            
            # y_pad = self._pad_y(y, task)                       # B × cond_dim
            h     = self.encoder(torch.cat([x, y_pad], dim=-1))
            mu, logvar = self.mu(h), self.logvar(h)
            z     = self._reparam(mu, logvar)
            recon = self.decoders[task](torch.cat([x, z], dim=-1)) 
            return recon, mu, logvar
        else:
            z = torch.randn(x.size(0), self.mu.out_features, device=x.device)
            recon = self.decoders[task](torch.cat([x, z], dim=-1))
            return recon
