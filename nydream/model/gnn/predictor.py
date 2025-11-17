from typing import List, Iterable, Union, Optional, Callable

import numpy as np
import rdkit.Chem.AllChem as Chem
import json

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import Linear


class EndToEndModule(nn.Module):
    def __init__(self, gnn_embedder: nn.Module, nn_predictor: nn.Module):
        super(EndToEndModule, self).__init__()
        self.gnn_embedder = gnn_embedder
        self.nn_predictor = nn_predictor
    
    def forward(self, data: pyg.data.Data,):
        embedding = self.gnn_embedder(data)
        output = self.nn_predictor(embedding)
        return output, embedding
            
