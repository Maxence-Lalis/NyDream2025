import torch
from training.make_train_step import make_train_step

def make_train_epoch(model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     train_loader,
                     loss_fn,
                     device: torch.device,
                     loss_fn2 = None,
                     lambda_val = 1.0,
                     alpha = None,
                     tid2task = None,
                     ):
    """
    Run one training epoch over `train_loader`.
    Returns the average training loss.
    """
    running = 0.0
    for batch in train_loader:
        step_loss = make_train_step(model, optimizer, batch, loss_fn, device, loss_fn2, lambda_val, alpha, tid2task)
        running += step_loss
    return running / max(1, len(train_loader))