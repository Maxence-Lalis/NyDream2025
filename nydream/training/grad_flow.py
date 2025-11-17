import collections
import torch

class GradFlowMonitor:
    """
    Tracks gradient norms via param.register_hook during backprop.
    By default, monitors FiLM params (using your is_film_param(name)).
    """
    def __init__(self, name_filter_fn):
        self.name_filter_fn = name_filter_fn
        self._step_totals = collections.defaultdict(float)   # sum of grad norms within current epoch
        self._step_counts = collections.defaultdict(int)     # number of hits within current epoch
        self._epoch_totals = collections.defaultdict(float)  # accumulates across steps automatically via hooks
        self._epoch_counts = collections.defaultdict(int)
        self._handles = []
        self._enabled = True

    def attach(self, model):
        # Register a hook on each parameter that passes the filter
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self.name_filter_fn(name):
                h = p.register_hook(self._make_hook(name))
                self._handles.append(h)
        return self

    def _make_hook(self, name):
        def _hook(grad):
            if not self._enabled or grad is None:
                return
            # L2 norm of gradient tensor
            g = grad.detach()
            # handle sparse grads (unlikely here)
            try:
                nrm = float(g.norm().item())
            except Exception:
                nrm = float(g.to_dense().norm().item())
            self._epoch_totals[name] += nrm
            self._epoch_counts[name] += 1
        return _hook

    def epoch_stats(self, reduce_grouped=True, reset=True):
        """
        Returns dict with mean grad norm per parameter (or grouped by module)
        Example keys (grouped): film.total, film.edge_mlp, film.node_mlp, ...
        """
        stats = {}
        if reduce_grouped:
            # group by top-level token around 'film' for readability
            grouped = collections.defaultdict(lambda: {"sum":0.0, "cnt":0})
            for name, tot in self._epoch_totals.items():
                key = self._group_key(name)
                grouped[key]["sum"] += tot
                grouped[key]["cnt"] += self._epoch_counts[name]
            for k, v in grouped.items():
                stats[k] = v["sum"] / max(1, v["cnt"])
            # also a global total
            total_sum = sum(v["sum"] for v in grouped.values())
            total_cnt = sum(v["cnt"] for v in grouped.values())
            stats["film.total"] = total_sum / max(1, total_cnt)
        else:
            for name, tot in self._epoch_totals.items():
                stats[name] = tot / max(1, self._epoch_counts[name])
        if reset:
            self._epoch_totals.clear()
            self._epoch_counts.clear()
        return stats

    def _group_key(self, name: str) -> str:
        """
        Make a compact label: tries to extract submodule path around 'film'
        e.g., 'gnn_embedder.blocks.2.film_edge.net.3.weight' -> 'film_edge'
        """
        parts = name.split(".")
        for i, p in enumerate(parts):
            if "film" in p:
                # return the token containing 'film' (e.g., film_edge or film_node)
                return p
        return "film.other"

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
