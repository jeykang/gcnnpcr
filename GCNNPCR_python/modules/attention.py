import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.dk = d_model//nhead
    def forward(self, query, key, value):
        B,Q,_=query.shape; _,K,_=key.shape
        Q_ = self.q(query).view(B,Q,self.nhead,self.dk).permute(0,2,1,3)
        K_ = self.k(key).view(B,K,self.nhead,self.dk).permute(0,2,3,1)
        V_ = self.v(value).view(B,K,self.nhead,self.dk).permute(0,2,1,3)
        scores = (Q_ @ K_)/self.dk**0.5
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V_).permute(0,2,1,3).reshape(B,Q,self.nhead*self.dk)
        return out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            ca = CrossAttention(d_model,nhead)
            ff = nn.Sequential(nn.Linear(d_model,d_model*4), nn.ReLU(), nn.Linear(d_model*4,d_model))
            self.layers.append(nn.ModuleList([ca,ff, nn.LayerNorm(d_model), nn.LayerNorm(d_model)]))
    def forward(self, x, context):
        for ca,ff,n1,n2 in self.layers:
            x2 = ca(x, context, context); x = n1(x+x2)
            x2 = ff(x); x = n2(x+x2)
        return x
