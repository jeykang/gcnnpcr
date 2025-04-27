# -*- coding: utf-8 -*- 

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
import random
import torch.nn.functional as F
from .emdloss_new import SinkhornEMDLoss

# ====================== Normals Utility ======================
def local_knn_coords(coords, k=16):
    dist = torch.cdist(coords, coords)
    knn_idx = dist.topk(k, largest=False, dim=1).indices
    return knn_idx

def robust_loadtxt(file_path):
    valid_rows = []
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                row_floats = [float(x) for x in parts[:6]]
                valid_rows.append(row_floats)
            except ValueError:
                continue
    return np.array(valid_rows)

def compute_normals_pca(coords_t, k=16):
    N = coords_t.shape[0]
    knn_idx = local_knn_coords(coords_t, k=k)
    normals = torch.zeros_like(coords_t)
    for i in range(N):
        neighbor_pts = coords_t[knn_idx[i]]
        mean_ = neighbor_pts.mean(dim=0, keepdim=True)
        cov_  = (neighbor_pts - mean_).t() @ (neighbor_pts - mean_)
        eigvals, eigvecs = torch.linalg.eigh(cov_)
        normal_i = eigvecs[:, 0]
        normals[i] = normal_i
    normals = F.normalize(normals, dim=-1)
    return normals

class S3DISDataset(Dataset):
    def __init__(self, 
                 root: str,
                 mask_ratio: float = 0.5,
                 num_points: int = 8192,
                 split: str = 'train',
                 normal_k: int = 16):
        super().__init__()
        self.root = root
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.split = split
        self.normal_k = normal_k

        pattern = os.path.join(root, '**', '*.txt')
        all_files = [f for f in glob(pattern, recursive=True)
                     if ('alignmentAngle' not in f and 'Annotations' not in f)]
        split_idx = int(0.8 * len(all_files))
        if self.split == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        arr = robust_loadtxt(file_path)
        if arr.shape[0] == 0:
            arr = np.zeros((self.num_points, 6), dtype=np.float32)

        N = arr.shape[0]
        if N < self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
        else:
            indices = np.random.choice(N, self.num_points, replace=False)

        sample = arr[indices]  # => [num_points, 6]
        coords = sample[:, :3]

        # normalize
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c)/2
        scale = (max_c - min_c).max()/2
        if scale < 1e-8:
            scale=1.0
        coords = (coords - center)/scale

        coords_t = torch.from_numpy(coords).float()
        # compute normals
        normals_t = compute_normals_pca(coords_t, k=self.normal_k)  # => [N,3]

        # combine => [N,6]
        full_6d = torch.cat([coords_t, normals_t], dim=-1)  # => [N,6]

        # mask
        mask_count = int(self.num_points * self.mask_ratio)
        mask_idx = np.random.choice(self.num_points, mask_count, replace=False)
        partial_6d = full_6d.clone()
        partial_6d[mask_idx, :] = 0.0  # zero out coords + normals

        return {
            "partial": partial_6d,  # [num_points,6]
            "full": full_6d        # [num_points,6]
        }

# ====================== Repulsion, random_resample, etc. ======================
import torch.nn.functional as F

def pairwise_dist(points):
    B, N, _ = points.shape
    xx = (points * points).sum(dim=-1, keepdim=True)
    dist = xx + xx.transpose(1,2) - 2 * torch.matmul(points, points.transpose(1,2))
    return F.relu(dist)

def repulsion_loss(pred_points, k=4, threshold=0.02):
    B, N, _ = pred_points.shape
    dist_mat = pairwise_dist(pred_points)
    knn_vals, _ = torch.topk(dist_mat, k=k+1, largest=False, dim=-1)
    knn_vals = knn_vals[..., 1:]
    rep = F.relu(threshold - knn_vals)
    return rep.mean()

def random_resample(points, out_n=8192):
    B, N, _ = points.shape
    if N <= out_n:
        idx = torch.randint(0, N, size=(B, out_n), device=points.device)
    else:
        idx = torch.stack([
            torch.randperm(N, device=points.device)[:out_n]
            for _ in range(B)
        ], dim=0)
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, 3)
    out_points = torch.gather(points, dim=1, index=idx_expand)
    return out_points

# ====================== GCN with in_dim=6 ======================
from torch_geometric.nn import knn_graph, GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dims=[64, 128], out_dim=128, k=16):
        super().__init__()
        self.k = k
        self.gconvs = nn.ModuleList()
        prev_dim = in_dim
        for hd in hidden_dims:
            self.gconvs.append(GCNConv(prev_dim, hd))
            prev_dim = hd
        self.final_lin = nn.Linear(prev_dim, out_dim)

    def forward(self, coords_batch):
        """
        coords_batch: [B,N,6]
        We'll use coords_batch[..., :3] for adjacency, 
        but keep all 6 dims as node features for GCN.
        returns: [B,N,out_dim]
        """
        B, N, C = coords_batch.shape
        device = coords_batch.device

        all_feats = []
        for b in range(B):
            feats_b = coords_batch[b]         # => [N,6]
            coords_3d = feats_b[:, :3]        # => [N,3] for adjacency
            edge_idx = knn_graph(coords_3d, k=self.k, loop=False)
            x = feats_b.clone()  # 6D as node features
            for conv in self.gconvs:
                x = conv(x, edge_idx)
                x = F.relu(x)
            x_out = self.final_lin(x)  # => [N,out_dim]
            all_feats.append(x_out.unsqueeze(0))
        token_feats = torch.cat(all_feats, dim=0) # => [B,N,out_dim]
        return token_feats

# ====================== Geometry-Aware Transformer ======================
class GeomAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.dk = d_model//nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, coords):
        """
        x: [B,N,d_model]
        coords: [B,N,3]
        """
        B, N, D = x.shape
        Q = self.w_q(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)
        K = self.w_k(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)
        V = self.w_v(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.dk**0.5) # => [B,nhead,N,N]

        # geometry bias with coords => [B,N,3]
        coords_2d = coords.view(B*N, 3)
        dot_mat = torch.matmul(coords_2d, coords_2d.T)  # => [B*N, B*N]
        G = torch.zeros((B, N, N), device=x.device)
        for b in range(B):
            st = b*N
            ed = (b+1)*N
            G[b] = dot_mat[st:ed, st:ed]
        G = self.alpha*G
        G = G.unsqueeze(1).expand(-1,self.nhead,-1,-1)
        scores += G

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # => [B,nhead,N,dk]
        out = out.permute(0,2,1,3).contiguous().view(B,N,self.nhead*self.dk)
        return out

class GeomMultiTokenTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GeomAttention(d_model, nhead)
            ff = nn.Sequential(
                nn.Linear(d_model, 4*d_model),
                nn.ReLU(True),
                nn.Linear(4*d_model, d_model)
            )
            self.layers.append(nn.ModuleList([attn, ff]))
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, coords):
        """
        x: [B,N,d_model]
        coords: [B,N,3]
        => returns [B,N,d_model]
        """
        for i, (attn, ffn) in enumerate(self.layers):
            x_attn = attn(x, coords)
            x = x + x_attn
            x = self.norm1[i](x)
            x_ff = ffn(x)
            x = x + x_ff
            x = self.norm2[i](x)
        return x

# ====================== Local KNN aggregator + 1D Deconv approach ======================

def local_knn(partial, predicted, k=8):
    """
    partial:   [B,M,6] or [B,M,>6], but we only use partial[..., :3] for distances
    predicted: [B,N,6] or [B,N,>6], we use predicted[..., :3] for aggregator
    returns => [B,N,k,3] neighbors in coords. 
    """
    device = partial.device
    B, M, c_p = partial.shape
    _, N, c_t = predicted.shape
    out_list = []
    for b in range(B):
        part_b = partial[b, :, :3]   # [M,3]
        pred_b = predicted[b, :, :3] # [N,3]
        dist = torch.cdist(pred_b, part_b, p=2) 
        knn_idx = dist.topk(k, largest=False, dim=1).indices
        neigh_b = []
        for i in range(N):
            row_idx = knn_idx[i]
            neigh_b.append(part_b[row_idx]) # => [k,3]
        neigh_b = torch.stack(neigh_b, dim=0) # => [N,k,3]
        out_list.append(neigh_b.unsqueeze(0))
    return torch.cat(out_list, dim=0)  # => [B,N,k,3]

class MLP_Res(nn.Module):
    """
    Residual MLP akin to SnowflakeNet, ensures expansions keep prior shape info.
    """
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        # x: [B, in_dim, n]
        sc = self.shortcut(x)
        out = self.conv2(F.relu(self.conv1(x))) + sc
        return out

def fps_subsample(pcd_xyz, out_n):
    """
    pcd_xyz: [B, N, 3], a naive farthest point sampling
    returns: [B, out_n, 3]
    (Pseudo, not optimized. For large N, you want a better method.)
    """
    # For demonstration, let's do random if we want to keep code short
    B, N, _ = pcd_xyz.shape
    device = pcd_xyz.device
    if N <= out_n:
        return pcd_xyz
    idx = torch.stack([torch.randperm(N, device=device)[:out_n] for _ in range(B)], dim=0)
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, 3)
    out = torch.gather(pcd_xyz, 1, idx_expand)
    return out

class SPDStage(nn.Module):
    """
    A single Snowflake-like refine stage:
     - merges partial + predicted? 
     - upsample feature => K_curr
     - bounding => tanh
    """
    def __init__(self, in_feat_dim=128, radius=1.0, bounding=True, up_factor=2, stage_idx=0):
        super().__init__()
        self.radius = radius
        self.bounding = bounding
        self.up_factor = up_factor
        self.stage_idx = stage_idx
        self.in_feat_dim = in_feat_dim

        # We'll define:
        # 1) aggregator mlp for pcd -> local feature
        self.mlp_pcd = nn.Conv1d(3, 64, 1)  # simplistic local MLP
        self.mlp_merge = nn.Conv1d(64 + in_feat_dim, in_feat_dim, 1) # merges local feat + old K

        # 2) a "feature upsampler" akin to ConvTranspose1d
        self.deconv_feat = nn.ConvTranspose1d(in_feat_dim, in_feat_dim, kernel_size=up_factor, stride=up_factor)
        # 3) a residual MLP for offset
        self.mlp_offset = MLP_Res(in_dim=in_feat_dim, hidden_dim=in_feat_dim, out_dim=in_feat_dim)
        self.conv_offset = nn.Conv1d(in_feat_dim, 3, 1)

    def forward(self, pcd_xyz, K_prev, global_feat, partial_xyz):
        """
        pcd_xyz:   [B, 3, N]   => current stage geometry
        K_prev:    [B, in_feat_dim, N], old displacement feature or None
        global_feat: [B, in_feat_dim, 1], the global code
        partial_xyz: [B, Np, 3], partial for merging
        returns:
          pcd_up:   [B, 3, N * up_factor]
          K_curr:   [B, in_feat_dim, N * up_factor]
        """

        B, _, N = pcd_xyz.shape

        # 0) Optionally unify partial + pcd and do fps => pcd_merged
        # or let's do a simpler approach: aggregator uses partial aggregator
        # We'll just do a local aggregator: mlp over pcd_xyz. Then concat global_feat
        local_feat = self.mlp_pcd(pcd_xyz)  # => [B,64,N]
        if K_prev is None:
            # instead of `K_prev = local_feat` => 64 ch
            # create a zero or some initial 128-ch tensor
            # with shape [B, 128, N] 
            B, c_l, N = local_feat.shape  # c_l=64
            K_prev = local_feat.new_zeros((B, self.in_feat_dim, N))  # => [B,128,N]

        # merge => cat( local_feat, K_prev, global_feat repeated )
        # simpler approach: cat( local_feat, K_prev ) => [B,64 + in_feat_dim, N]
        #print("local_feat shape:", local_feat.shape)  # e.g. [B, 64, N]
        #print("K_prev shape:", K_prev.shape)          # e.g. [B, 64, N]
        feats_merge = torch.cat([local_feat, K_prev], dim=1)
        #print("feats_merge shape:", feats_merge.shape) # e.g. [B, 128, N]

        feats_merge = self.mlp_merge(feats_merge)  # => error if in_channels mismatch


        # inject global feature
        # broadcast => [B, in_feat_dim, N]
        gf_broad = global_feat.repeat(1,1,N)                 # => [B, in_feat_dim, N]
        feats_merge = feats_merge + gf_broad                 # => a simple add

        # 1) Up-sample K => [B, in_feat_dim, N*up_factor]
        K_up = self.deconv_feat(feats_merge)  # => [B,in_feat_dim,N*up_factor]
        # 2) offset MLP => do a residual mlp => [B,in_feat_dim,N*up_factor]
        offset_feat = self.mlp_offset(K_up)   # => [B,in_feat_dim,N*up_factor]

        # 3) final offset => [B,3,N*up_factor]
        delta = self.conv_offset(offset_feat)

        # bounding => tanh(...) / radius^stage_idx
        if self.bounding:
            delta = torch.tanh(delta) / (self.radius**(self.stage_idx))

        # up-sample geometry => tile pcd_xyz across up_factor
        # simplest approach => repeat_interleave
        # shape => [B,3,N*up_factor]
        pcd_up = pcd_xyz.repeat_interleave(self.up_factor, dim=2)

        # new pcd => pcd_up + delta
        pcd_up = pcd_up + delta
        # K_curr => offset_feat
        K_curr = offset_feat
        return pcd_up, K_curr



class SPDBasedDecoder(nn.Module):
    """
    3 SPD-like stages => after each stage, we up-factor => final ~ 8192
    bounding expansions, skip K features
    """
    def __init__(self, in_feat_dim=128, coarse_num=64, up_factors=[2,2,2,2,2,2,2], bounding=True, radius=1.0):
        super().__init__()
        self.coarse_num = coarse_num
        self.init_fc = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim),
            nn.ReLU(True),
            nn.Linear(in_feat_dim, 3*coarse_num)
        )
        # define stages
        self.stages = nn.ModuleList()
        stage_idx = 0
        curr = coarse_num
        for uf in up_factors:
            spd_stage = SPDStage(in_feat_dim=in_feat_dim, radius=radius, bounding=bounding, up_factor=uf, stage_idx=stage_idx)
            self.stages.append(spd_stage)
            stage_idx += 1

    def forward(self, partial_6d, global_feat):
        """
        partial_6d: [B, N, 6], for aggregator
        global_feat: [B, in_feat_dim, 1]
        """
        B, N, _ = partial_6d.shape
        # 0) produce coarse => [B,3,64]
        # global_feat: [B,in_feat_dim,1]
        fc_out = self.init_fc(global_feat.squeeze(-1))  # => [B,3*coarse_num]
        pcd_coarse = fc_out.view(B,3,self.coarse_num)

        # let K=None initially
        K = None
        pcd = pcd_coarse
        for stage_i, spd in enumerate(self.stages):
            # we do a partial aggregator or skip for brevity
            # if you want to unify partial+ pcd -> fps_subsample, do it here
            pcd, K = spd(pcd, K, global_feat, partial_6d[..., :3])

        return pcd  # final e.g. 8192


class FullModelSnowflake(nn.Module):
    """
    partial_6d => GraphEncoder => tokens => geometry-aware transformer => 
    => global_feat => SPD-based 3-stage decode => final 8192
    """
    def __init__(self,
                 g_hidden_dims=[64,128],
                 g_out_dim=128,
                 t_d_model=128,
                 t_nhead=8,
                 t_layers=4,
                 coarse_num=64,
                 bounding=True,
                 radius=1.0,
                 up_factors=[2,2,2,2,2,2,2]):
        super().__init__()
        self.graph_enc = GraphEncoder(in_dim=6, hidden_dims=g_hidden_dims, out_dim=g_out_dim)
        self.transformer = GeomMultiTokenTransformer(d_model=t_d_model, nhead=t_nhead, num_layers=t_layers)
        self.bridge = nn.Linear(g_out_dim, t_d_model) if g_out_dim!=t_d_model else nn.Identity()
        self.decoder = SPDBasedDecoder(in_feat_dim=t_d_model,
                                       coarse_num=coarse_num,
                                       up_factors=up_factors,
                                       bounding=bounding,
                                       radius=radius)

    def forward(self, partial_6d):
        B, N, _ = partial_6d.shape
        # 1) encode => [B,N,g_out_dim]
        tokens = self.graph_enc(partial_6d)
        # 2) transform => [B,N,t_d_model], coords => partial_6d[..., :3]
        x = self.bridge(tokens)
        x_out = self.transformer(x, partial_6d[..., :3])
        # 3) global => mean => [B,t_d_model,1]
        global_feat = x_out.mean(dim=1, keepdim=True).permute(0,2,1) # => [B,t_d_model,1]
        # 4) decode => e.g. 8192 final
        final_pcd = self.decoder(partial_6d, global_feat)
        return final_pcd

# ====================== Train Loop ======================
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, os
from tqdm import tqdm

def save_point_cloud_comparison(partial, completed, original, epoch, out_dir="visuals"):
    #partial:   [N, 3]  PyTorch tensor (the masked, incomplete input)
    #completed: [M, 3]  PyTorch tensor (the network�fs output)
    #original:  [N, 3]  PyTorch tensor (the unmasked, full point cloud)
    #Saves an image 'completion_epoch_{epoch}.png' with 3 side-by-side subplots.
    
    os.makedirs(out_dir, exist_ok=True)

    # Convert to NumPy
    partial_np   = partial.detach().cpu().numpy()
    completed_np = completed.detach().cpu().numpy()
    original_np  = original.detach().cpu().numpy()

    # -----------------------------------------------------------
    # 1) Compute a unified bounding box so the scales match
    # -----------------------------------------------------------
    all_points = np.concatenate([partial_np, completed_np, original_np], axis=0)
    min_xyz = all_points.min(axis=0)  # [3]
    max_xyz = all_points.max(axis=0)  # [3]

    # tiny epsilon if min==max
    eps = 1e-5
    range_xyz = max_xyz - min_xyz
    range_xyz[range_xyz < eps] = eps  # avoid singular transforms

    # -----------------------------------------------------------
    # 2) Set up the figure
    # -----------------------------------------------------------
    fig = plt.figure(figsize=(15, 5))

    # -----------------------------------------------------------
    # Subplot 1: Partial
    # -----------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=1, c='r')
    ax1.set_title("Partial (Masked)")
    ax1.set_xlim3d(min_xyz[0], max_xyz[0])
    ax1.set_ylim3d(min_xyz[1], max_xyz[1])
    ax1.set_zlim3d(min_xyz[2], max_xyz[2])

    # -----------------------------------------------------------
    # Subplot 2: Completed
    # -----------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(completed_np[:, 0], completed_np[:, 1], completed_np[:, 2], s=1, c='b')
    ax2.set_title("Completed (Predicted)")
    ax2.set_xlim3d(min_xyz[0], max_xyz[0])
    ax2.set_ylim3d(min_xyz[1], max_xyz[1])
    ax2.set_zlim3d(min_xyz[2], max_xyz[2])

    # -----------------------------------------------------------
    # Subplot 3: Original (Full)
    # -----------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(original_np[:, 0], original_np[:, 1], original_np[:, 2], s=1, c='g')
    ax3.set_title("Original (Unmasked)")
    ax3.set_xlim3d(min_xyz[0], max_xyz[0])
    ax3.set_ylim3d(min_xyz[1], max_xyz[1])
    ax3.set_zlim3d(min_xyz[2], max_xyz[2])

    # -----------------------------------------------------------
    # 3) Save and close
    # -----------------------------------------------------------
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"completion_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved point cloud comparison to {save_path}")


from fvcore.nn import FlopCountAnalysis
def train_s3dis_model():
    # dataset
    train_dataset = S3DISDataset(root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis", mask_ratio=0.5, num_points=8192, split='train', normal_k=16)
    val_dataset   = S3DISDataset(root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis", mask_ratio=0.5, num_points=8192, split='val',   normal_k=16)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModelSnowflake(
        g_hidden_dims=[64,128],
        g_out_dim=128,
        t_d_model=128,
        t_nhead=8,
        t_layers=4,
        coarse_num=64,
        radius=1.0
    ).to(device)

    #calc num params & flops
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    #flops = FlopCountAnalysis(model, input)
    #print("Number of flops:", flops.total())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        # Create an instance of our EMD loss
        # With suggested parameters
        emd_criterion = SinkhornEMDLoss(
            reg=0.05,        # try a slightly bigger reg than 0.01 for 8192
            max_iters=25,    # fewer iterations for speed
            num_samples=1024,# downsample from 8192->1024
            use_warm_start=True
        ).to(device)
        epoch_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(train_iter):
            partial_6d = batch["partial"].to(device)  # [B,8192,6]
            full_6d    = batch["full"].to(device)     # [B,8192,6]

            optimizer.zero_grad()
            completed = model(partial_6d)  # => [B,3,8192]
            completed_renorm = completed.permute(0, 2, 1).contiguous()  # => [B, 8192, 3]


            # we only have coords for "full" shape => full_6d[..., :3]
            # likewise for partial_6d we only do . but we do the final shape => [B,8192,3]
            # compute EMD
            loss_emd = emd_criterion(completed_renorm, full_6d[..., :3])
            cd_val, _ = chamfer_distance(completed_renorm, full_6d[..., :3])
            rep_val = repulsion_loss(completed_renorm, k=4, threshold=0.02)
            cd_coef = 1
            rep_coef = 0.1
            loss = loss_emd + cd_coef*cd_val + rep_coef*rep_val

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_iter.set_postfix({"loss": loss.item()})

        epoch_loss_avg = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss_avg:.4f}")

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_val in val_loader:
                partial_val_6d = batch_val["partial"].to(device)
                full_val_6d    = batch_val["full"].to(device)
                completed_val = model(partial_val_6d)
                completed_val_renorm = completed_val.permute(0, 2, 1).contiguous()  # => [B, 8192, 3]
                cd_val_v, _ = chamfer_distance(completed_val_renorm, full_val_6d[..., :3])
                val_loss += cd_val_v.item()
        val_loss_avg = val_loss / len(val_loader)
        print(f"    Validation Chamfer: {val_loss_avg:.4f}")

        # save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")

        # Example vis
        partial_0   = partial_6d[0,...,:3]  # first 3 dims for plotting
        completed_0 = completed_renorm[0]
        original_0  = full_6d[0,...,:3]
        save_point_cloud_comparison(partial_0, completed_0, original_0, epoch+1)

    return model


if __name__ == "__main__":
    train_s3dis_model()
