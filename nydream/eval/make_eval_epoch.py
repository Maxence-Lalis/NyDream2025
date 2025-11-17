import torch
from eval.make_eval_step import make_eval_step
from collections import defaultdict

@torch.no_grad()
def make_eval_epoch(model: torch.nn.Module,
                    test_loader,
                    loss_fn,
                    metric_fn,
                    device: torch.device,
                    loss_fn2=None,
                    lambda_val=1,
                    tid2task=None):

    def _mean(vals):
        return sum(vals) / len(vals) if len(vals) else float("nan")

    model.eval()

    # Collect predictions/targets per task
    buckets_pred = defaultdict(list)
    buckets_true = defaultdict(list)

    for batch in test_loader:
        out = make_eval_step(model, batch, device, tid2task)  # {task_name: (y_pred_cpu, y_true_cpu)}
        for task_name, (yp, yt) in out.items():
            buckets_pred[task_name].append(yp)
            buckets_true[task_name].append(yt)

    # Concatenate within each task only
    per_task_pred = {t: torch.cat(buckets_pred[t], dim=0) for t in buckets_pred}
    per_task_true = {t: torch.cat(buckets_true[t], dim=0) for t in buckets_true}

    per_task_report = {}
    per_task_losses = []
    per_task_metrics = []  # list of scalars or dicts
    per_task_ns = []       # number of samples per task (NOT elements)

    # ---- Per-task computation using loss_fn / metric_fn directly ----
    for task in per_task_pred:
        y_pred = per_task_pred[task]
        y_true = per_task_true[task]

        # (safety) match width if needed
        if y_pred.ndim > 1 and y_true.ndim > 1 and y_true.size(1) != y_pred.size(1):
            y_true = y_true[..., :y_pred.size(1)]

        N = y_true.size(0)
        per_task_ns.append(N)

        # Loss (same reduction as your loss_fn; typically 'mean')
        loss = loss_fn(y_pred.squeeze(), y_true.squeeze())
        if loss_fn2 is not None:
            y_prob = torch.sigmoid(y_pred)
            loss = loss + lambda_val * loss_fn2(y_prob, y_true.squeeze())
        loss_value = float(loss.item())
        per_task_losses.append(loss_value)

        # Metrics via metric_fn logic
        if metric_fn is None:
            metric_value = None
        elif isinstance(metric_fn, dict):
            metric_value = {}
            for name, fn in metric_fn.items():
                m = fn(y_pred.squeeze(), y_true.squeeze())
                metric_value[name] = float(m) if isinstance(m, (float, int)) else float(m.item())
        else:
            m = metric_fn(y_pred.squeeze(), y_true.squeeze())
            metric_value = float(m) if isinstance(m, (float, int)) else float(m.item())
        per_task_metrics.append(metric_value)

        per_task_report[task] = {"loss": loss_value, "metrics": metric_value}

    # ---- Macro (unweighted across tasks) ----
    macro_loss = _mean(per_task_losses) if per_task_losses else float("nan")
    if per_task_metrics and isinstance(per_task_metrics[0], dict):
        macro_metrics = {k: _mean([m[k] for m in per_task_metrics]) for k in per_task_metrics[0].keys()}
    elif per_task_metrics and per_task_metrics[0] is not None:
        macro_metrics = _mean(per_task_metrics)
    else:
        macro_metrics = None

    # ---- Global micro (sample-weighted average across tasks; no concat) ----
    total_N = sum(per_task_ns) if per_task_ns else 0
    if total_N == 0:
        global_micro_loss = float("nan")
        if per_task_metrics and isinstance(per_task_metrics[0], dict):
            global_micro_metrics = {k: float("nan") for k in per_task_metrics[0].keys()}
        elif per_task_metrics:
            global_micro_metrics = float("nan")
        else:
            global_micro_metrics = None
    else:
        # loss
        global_micro_loss = sum(l * N for l, N in zip(per_task_losses, per_task_ns)) / total_N
        # metrics
        if per_task_metrics and isinstance(per_task_metrics[0], dict):
            keys = per_task_metrics[0].keys()
            global_micro_metrics = {k: sum(m[k] * N for m, N in zip(per_task_metrics, per_task_ns)) / total_N for k in keys}
        elif per_task_metrics and per_task_metrics[0] is not None:
            global_micro_metrics = sum(m * N for m, N in zip(per_task_metrics, per_task_ns)) / total_N
        else:
            global_micro_metrics = None

    metrics_dict = {
        "per_task": per_task_report,
        "macro_avg": {"loss": macro_loss, "metrics": macro_metrics},
        "global_micro": {"loss": global_micro_loss, "metrics": global_micro_metrics},
    }
    # Keep original API: first return is the overall (micro) loss
    return global_micro_loss, metrics_dict


# def make_eval_epoch(model: torch.nn.Module,
#                     test_loader,
#                     loss_fn,
#                     metric_fn,
#                     device: torch.device,
#                     loss_fn2=None,
#                     lambda_val=1,
#                     tid2task=None):
#     """
#     Evaluate the model on `test_loader` (single-dataset setup).

#     Computes the loss on the concatenated predictions/targets to match
#     "compute-once" semantics (not per-batch averaging).

#     Returns:
#         (loss_value, metric_value or None)
#     """
#     model.eval()
#     y_pred, y_true = [], []

#     for batch in test_loader:
#         y_hat_b, y_b = make_eval_step(model, batch, device, tid2task)
#         y_pred.append(y_hat_b)
#         y_true.append(y_b)

#     y_pred = torch.cat(y_pred, dim=0)
#     y_true = torch.cat(y_true, dim=0)

#     loss = loss_fn(y_pred.squeeze(), y_true.squeeze())
#     if loss_fn2 is not None:
#         preds = torch.sigmoid(y_pred)
#         loss = loss + lambda_val * loss_fn2(preds, y_true.squeeze())

#     loss_value = float(loss.item())

#     metric_value  = None
#     if metric_fn is not None:
#         # When a dictionary of metric functions is provided, compute each one
#         if isinstance(metric_fn, dict):
#             metric_value = {}
#             for name, fn in metric_fn.items():
#                 m = fn(y_pred.squeeze(), y_true.squeeze())
#                 # Flatten any tensor values to plain floats
#                 metric_value[name] = float(m) if isinstance(m, (float, int)) else float(m.item())
#         else:
#             m = metric_fn(y_pred.squeeze(), y_true.squeeze())
#             metric_value = float(m) if isinstance(m, (float, int)) else float(m.item())

#     return loss_value, metric_value
