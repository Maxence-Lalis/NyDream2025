import torch

def make_train_step(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    batch,
                    loss_fn,
                    device: torch.device,
                    loss_fn2 = None,
                    lambda_val = 1.0,
                    alpha = None,
                    tid2task = None,
                    ):
    """
    Perform a single optimization step on one batch.
    """
    model.train()
    
    # Phase A/B
    if len(batch)==2:
        data, y = batch
        data, y = data.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat, _ = model(data)

        loss = loss_fn(y_hat.squeeze(), y.squeeze())
        if loss_fn2 is not None:
            preds = torch.sigmoid(y_hat)
            loss = loss + lambda_val * loss_fn2(preds, y.squeeze())

        loss.backward()
        optimizer.step()
    
    # Phase C
    elif len(batch)==3:
        data, y, tid = batch
        tid = tid.squeeze(1) 
        data, y, tid = data.to(device), y.to(device), tid.to(device)
        
        optimizer.zero_grad()
        loss = torch.zeros(1, device=device)

        for t_idx in tid.unique():
            m = (tid == t_idx)
            task_name = tid2task[int(t_idx.item())]

            y_hat = model(data[m], task=task_name)
            out_dim = y_hat.size(1)
            y_task = y[m][..., :out_dim]                   

            rec = loss_fn(y_hat, y_task)

            if loss_fn2 is not None:
                preds = torch.sigmoid(y_hat)
                loss = rec + lambda_val * loss_fn2(preds, y_task)

            loss += alpha[t_idx] * rec

        loss.backward()
        optimizer.step()

    return float(loss.item())