from typing import List, Tuple
import torch
from torch import nn

from make_inference_step import make_inference_step

def make_inference_epoch(model, 
                         inference_loader, 
                         device,
                         taskname = None):
    """
    Iterate over a DataLoader and return concatenated predictions (CPU tensor).
    Parameters
    ----------
    model : nn.Module
    End-to-end model with signature: output, embedding = model(batch)
    loader : torch.utils.data.DataLoader
    DataLoader yielding torch_geometric.data.Batch instances.
    device : str
    Target device string.


    Returns
    -------
    torch.Tensor
    Tensor of shape [N, out_dim] with predictions for all samples.
    """
    
    model.eval()
    preds: List[torch.Tensor] = []
    embeds: List[torch.Tensor] = []
    
    for batch in inference_loader:
        y_pred, embedding = make_inference_step(model, batch, device, taskname)
        preds.append(y_pred.detach().cpu())
        if embedding is not None:
            embeds.append(embedding.detach().cpu())
            return torch.concat(preds, dim=0), torch.concat(embeds, dim=0)
        else:
            return torch.concat(preds, dim=0), None