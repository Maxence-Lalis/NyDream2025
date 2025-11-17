import torch
from torch import nn
from typing import Tuple


def make_inference_step(model, batch, device, taskname):
    """Run a forward pass on one batch and return predictions on CPU.

    Parameters
    ----------
    model : nn.Module
    End-to-end model with signature: output, embedding = model(batch)
    batch : torch_geometric.data.Batch
    Batched graph data.
    device : str
    Target device (e.g., "cpu", "cuda:0").


    Returns
    -------
    torch.Tensor
    Predictions on CPU with shape [B, out_dim].
    """
    model.eval()
    with torch.no_grad():
        data = batch
        data = data.to(device)
        if taskname:
            y_pred = model(data, task=taskname)
            return y_pred.detach().cpu(), None
        else:
            y_pred, embeddings = model(data)
            return y_pred.detach().cpu(), embeddings.detach().cpu()