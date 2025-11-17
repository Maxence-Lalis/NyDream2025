import torch

def get_loss_fn(loss_type):
    
    if loss_type == "MSE":
        loss_fn = torch.nn.MSELoss()
        return loss_fn
    
    elif loss_type == "BCEWithLogitsLoss":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn
    
    elif loss_type == "HuberLoss":
        loss_fn = torch.nn.HuberLoss()
        return loss_fn

    else:
        raise NotImplementedError