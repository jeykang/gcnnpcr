import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class DGCNNEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dims=[64,128], out_dim=128, k=16):
        super().__init__()
        mlps = []
        prev=in_dim
        for h in hidden_dims:
            mlps.append(nn.Sequential(nn.Linear(2*prev,h), nn.ReLU()))
            prev=h
        self.convs = nn.ModuleList([DynamicEdgeConv(mlp, k) for mlp in mlps])
        self.fc = nn.Linear(prev, out_dim)
    def forward(self, feats, coords):
        # feats: [B,N,in_dim]
        x = feats
        for conv in self.convs:
            x = conv(x, coords)
        # global pool
        return self.fc(x.max(dim=1)[0])  # [B,out_dim]
