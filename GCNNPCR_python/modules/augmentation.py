import torch

def jitter(x, sigma=0.01, clip=0.05):
    noise = x.new_empty(x.shape).normal_(0, sigma).clamp(-clip,clip)
    return x + noise

def scale(x, factor=0.9):
    s = x.new_empty(x.shape[0],1,1).uniform_(factor,1/factor)
    return x * s

def occlude(x, drop_ratio=0.1):
    B,N,C = x.shape
    m = int(N*drop_ratio)
    idx = torch.randperm(N)[:m]
    x[:,idx,:] = 0
    return x

def augment_pointcloud(pc):
    # pc: [B,N,6], first 3 dims are coords
    coords, feats = pc[...,:3], pc[...,3:]
    coords = jitter(coords)
    coords = scale(coords)
    coords = occlude(coords)
    return torch.cat([coords, feats], dim=-1)
