import torch
import numpy as np

def _multilabel_pearson():
    """Mean Pearson‑r across label columns (ignores constant columns)."""
    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        targets = y_true.detach().cpu().numpy()
        rs = []
        for col in range(targets.shape[1]):
            # Skip columns with no variance (Pearson is undefined)
            if targets[:, col].std() == 0:
                continue
            rs.append(np.corrcoef(probs[:, col], targets[:, col])[0, 1])
        return float(np.nanmean(rs)) if rs else np.nan
    return _metric

def _multilabel_f1():
    """Mean F1‑score across label columns using a 0.5 threshold."""
    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        targets = y_true.detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        scores = []
        for col in range(targets.shape[1]):
            pred_col = preds[:, col]
            true_col = targets[:, col]
            tp = np.logical_and(pred_col == 1, true_col == 1).sum()
            fp = np.logical_and(pred_col == 1, true_col == 0).sum()
            fn = np.logical_and(pred_col == 0, true_col == 1).sum()
            denom = 2 * tp + fp + fn
            if denom == 0:
                # Avoid division by zero for rarely positive labels
                continue
            f1 = 2 * tp / denom
            scores.append(f1)
        return float(np.nanmean(scores)) if scores else np.nan
    return _metric

def _multilabel_cosine():
    """Mean cosine similarity between predicted probabilities and binary targets."""
    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        targets = y_true.detach().cpu().numpy()
        dot = (probs * targets).sum(axis=1)
        norm_pred = np.linalg.norm(probs, axis=1)
        norm_true = np.linalg.norm(targets, axis=1)
        cosines = dot / (norm_pred * norm_true + 1e-8)
        return float(np.nanmean(cosines)) if cosines.size > 0 else np.nan
    return _metric

def _per_sample_cosine():
    """Mean cosine similarity between predicted probabilities and binary targets."""
    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        probs = y_pred.detach().cpu().numpy()
        targets = y_true.detach().cpu().numpy()
        dot = (probs * targets).sum(axis=1)
        norm_pred = np.linalg.norm(probs, axis=1)
        norm_true = np.linalg.norm(targets, axis=1)
        cosines = dot / (norm_pred * norm_true + 1e-8)
        return float(np.nanmean(cosines)) if cosines.size > 0 else np.nan
    return _metric

def _per_sample_pearson():
    """
    Mean Pearson-r across samples, where each sample's r is computed across label columns.
    Skips samples whose targets have zero variance.
    """
    def _metric(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        y_pred = torch.sigmoid(y_pred)
        yp = y_pred.detach()
        yt = y_true.detach()
        yp_c = yp - yp.mean(dim=1, keepdim=True)
        yt_c = yt - yt.mean(dim=1, keepdim=True)
        denom = (yp_c.norm(dim=1) * yt_c.norm(dim=1))

        valid = denom > 1e-9 
        if not torch.any(valid):
            return float('nan')
        r = (yp_c[valid] * yt_c[valid]).sum(dim=1) / denom[valid]  
        return float(r.mean().cpu())

    return _metric

def get_metric_fn(metric_type):
    """
    Return metric function(s) for multilabel classification.

    * If ``metric_type`` is the name of a single metric (e.g. ``"multilabel_pearson"``)
      the corresponding callable is returned.
    * Passing ``"multilabel_metrics"`` yields a dictionary of all available metrics
      (Pearson, F1 and cosine).  This makes it easy to compute several metrics
      concurrently.
    * Alternatively, supply an iterable of metric names to receive just those in a
      dictionary.
    """
    metrics = {
        "multilabel_pearson": _multilabel_pearson(),
        "multilabel_f1": _multilabel_f1(),
        "multilabel_cosine": _multilabel_cosine(),
        "sample_pearson":_per_sample_pearson(),
        "sample_cosine":_per_sample_cosine()
    }

    if isinstance(metric_type, str):
        if metric_type == "multilabel_metrics":
            return metrics
        if metric_type not in metrics:
            raise ValueError(f"Unsupported metric_type: {metric_type}")
        return metrics[metric_type]

    # Assume any other input is an iterable of names
    return {name: metrics[name] for name in metric_type if name in metrics}