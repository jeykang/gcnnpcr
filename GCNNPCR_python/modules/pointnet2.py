import torch
import torch.nn as nn
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

class SAModule(nn.Module):
    def __init__(self, ratio, r, mlp):
        super().__init__()
        self.ratio, self.r = ratio, r
        self.conv = PointConv(mlp)
    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos, self.r, batch, batch, max_num_neighbors=64)
        x = self.conv(x, (pos, pos), (row, col))
        return x[idx], pos[idx], batch[idx]

class PointNet2Encoder(nn.Module):
    def __init__(self, in_dim=6, out_dim=128):
        super().__init__()
        self.sa1 = SAModule(0.5, 0.2,
            nn.Sequential(nn.Linear(in_dim+3,64), nn.ReLU(), nn.Linear(64,64)))
        self.sa2 = SAModule(0.25, 0.4,
            nn.Sequential(nn.Linear(64+3,128), nn.ReLU(), nn.Linear(128,128)))
        self.lin = nn.Linear(128, out_dim)
    def forward(self, feats, coords):
        # feats: [B,N,in_dim], coords: [B,N,3]
        B,N,_ = feats.shape
        x = feats.view(B*N, -1)
        pos = coords.view(B*N, 3)
        batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x = global_max_pool(x, batch)  # [B,128]
        return self.lin(x)             # [B,out_dim]
