import torch 

@torch.no_grad()
def make_eval_step(model,
                   batch,
                   device,
                   tid2task):
    """
    Run a forward pass on one batch and return predictions and targets on CPU.

    Returns:
        (y_pred_cpu, y_true_cpu)
    """
    model.eval()
    
    if len(batch)==2:
        data, y = batch
        data = data.to(device)

        y_hat, _ = model(data)
        return {"__all__": (y_hat.detach().cpu(), y.detach().cpu())}
    
    elif len(batch) == 3:
        data, y, tid = batch
        tid = tid.squeeze(1)  # [B]
        data, y, tid = data.to(device), y.to(device), tid.to(device)

        per_task = {}
        # Ensure stable order
        for t_idx in tid.unique(sorted=True).tolist():
            m = (tid == t_idx)
            task_name = tid2task[int(t_idx)] if (tid2task is not None and int(t_idx) in tid2task) else f"task_{int(t_idx)}"

            # Forward only the subset for this task
            y_hat = model(data[m], task=task_name)
            out_dim = y_hat.size(1)
            y_true_task = y[m][..., :out_dim]

            per_task[task_name] = (y_hat.detach().cpu(), y_true_task.detach().cpu())

        return per_task    