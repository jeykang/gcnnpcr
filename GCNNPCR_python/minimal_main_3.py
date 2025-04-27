import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import random
from emdloss import emd_loss_sinkhorn

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

class S3DISDataset(Dataset):
    def __init__(self, 
                 root: str,
                 mask_ratio: float = 0.5,
                 num_points: int = 4096,
                 split: str = 'train'):
        super().__init__()
        self.root = root
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.split = split

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

        # sample exactly num_points
        N = arr.shape[0]
        if N < self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
        else:
            indices = np.random.choice(N, self.num_points, replace=False)

        sample = arr[indices]  # => [num_points, 6]
        coords = sample[:, :3] # => [num_points,3]

        # normalize to [-1,1]
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c)/2
        scale = (max_c - min_c).max()/2
        if scale < 1e-8:
            scale=1.0
        coords = (coords - center)/scale

        # randomly mask out mask_ratio fraction => set coords to zero
        mask_count = int(self.num_points * self.mask_ratio)
        mask_idx = np.random.choice(self.num_points, mask_count, replace=False)
        partial_coords = coords.copy()
        partial_coords[mask_idx] = 0.0

        partial_t = torch.from_numpy(partial_coords).float()
        full_t    = torch.from_numpy(coords).float()
        return {
            "partial": partial_t,  # [num_points,3]
            "full": full_t         # [num_points,3]
        }

##############################################################################
# 2) Optional: Repulsion utility
##############################################################################
def pairwise_dist(points):
    B, N, _ = points.shape
    xx = (points * points).sum(dim=-1, keepdim=True)
    dist = xx + xx.transpose(1,2) - 2 * torch.matmul(points, points.transpose(1,2))
    return F.relu(dist)

def repulsion_loss(pred_points, k=4, threshold=0.02):
    B, N, _ = pred_points.shape
    dist_mat = pairwise_dist(pred_points)
    knn_vals, _ = torch.topk(dist_mat, k=k+1, dim=-1, largest=False)
    knn_vals = knn_vals[..., 1:]
    rep = F.relu(threshold - knn_vals)
    return rep.mean()

def random_resample(points, out_n=4096):
    """
    points: [B, N, 3], want [B, out_n, 3]
    We pick 'out_n' random indices from N.
    """
    B, N, _ = points.shape
    if N <= out_n:
        # If already fewer points than out_n, just pad or replicate.
        # For demonstration, we replicate:
        idx = torch.randint(0, N, size=(B, out_n), device=points.device)
    else:
        idx = torch.stack([
            torch.randperm(N, device=points.device)[:out_n]
            for _ in range(B)
        ], dim=0)  # [B, out_n]
    # gather
    # We expand an index for the last dimension
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, 3)  # [B, out_n, 3]
    out_points = torch.gather(points, dim=1, index=idx_expand)
    return out_points

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, GCNConv  # example

class GraphEncoder(nn.Module):
    """
    A GCN-based encoder that transforms partial points -> multi_token embeddings [B,N,d_model].
    We'll treat each partial as a graph with KNN edges, run GCN layers, output a feature for each point => multi tokens.
    """
    def __init__(self, in_dim=3, hidden_dims=[64, 128], out_dim=128, k=16):
        super().__init__()
        self.k = k
        self.gconvs = nn.ModuleList()
        self.num_layers = len(hidden_dims)
        prev_dim = in_dim
        for hd in hidden_dims:
            self.gconvs.append(GCNConv(prev_dim, hd))
            prev_dim = hd
        self.final_lin = nn.Linear(prev_dim, out_dim)

    def forward(self, coords_batch):
        """
        coords_batch: [B, N, 3]
        returns: token_feats => [B, N, out_dim], i.e. multi-token
        We'll build adjacency per sample in the batch (loop).
        """
        B, N, C = coords_batch.shape
        device = coords_batch.device

        # We'll flatten across batch dimension, then build a single graph with offset.
        # E.g. we do a block approach: for b in [0..B-1], those belong to the offset range.
        # Or we do a loop. For demonstration, let's do a loop approach, then combine.

        all_feats = []
        for b in range(B):
            coords_b = coords_batch[b]  # => [N,3]
            # build KNN graph
            edge_idx = knn_graph(coords_b, k=self.k, loop=False)
            # We'll use coords_b as x for GCN
            x = coords_b.clone()  # [N,3]
            # Pass through GCN layers
            for conv in self.gconvs:
                x = conv(x, edge_idx)
                x = F.relu(x)
            # final
            x_out = self.final_lin(x)  # => [N,out_dim]
            all_feats.append(x_out.unsqueeze(0)) # => [1,N,out_dim]
        token_feats = torch.cat(all_feats, dim=0) # => [B,N,out_dim]
        return token_feats

class GeomMultiTokenTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        # We'll do a repeated block
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
        returns: [B,N,d_model]
        """
        for i, (attn, ffn) in enumerate(self.layers):
            x_attn = attn(x, coords)
            x = x + x_attn
            x = self.norm1[i](x)
            x_ff = ffn(x)
            x = x + x_ff
            x = self.norm2[i](x)
        return x

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
        B, N, D = x.shape
        Q = self.w_q(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)
        K = self.w_k(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)
        V = self.w_v(x).view(B, N, self.nhead, self.dk).permute(0,2,1,3)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.dk**0.5) # => [B,nhead,N,N]

        # geometry bias G: [B,N,N]
        coords_2d = coords.view(B*N, 3)
        dot_mat = torch.matmul(coords_2d, coords_2d.T)  # => [B*N, B*N]
        # block diag
        G = torch.zeros((B, N, N), device=x.device)
        for b in range(B):
            st = b*N
            ed = (b+1)*N
            G[b] = dot_mat[st:ed, st:ed]
        G = self.alpha*G
        G = G.unsqueeze(1).expand(-1,self.nhead,-1,-1)  # => [B,nhead,N,N]
        scores += G

        attn = F.softmax(scores, dim=-1)  # => [B,nhead,N,N]
        out = torch.matmul(attn, V)       # => [B,nhead,N,dk]
        out = out.permute(0,2,1,3).contiguous().view(B,N,self.nhead*self.dk)
        return out


def local_knn(partial, predicted, k=8):
    """
    partial:   [B,M,3]
    predicted: [B,N,3]
    For each predicted point, we find indices of k nearest partial points.
    returns partial_neigh => [B,N,k,3]
    """
    # We'll do a naive BFS approach: (B*N, M) => O(B*N*M). For large N, consider a more efficient method.
    device = partial.device
    B, M, _ = partial.shape
    _, N, _ = predicted.shape
    out = []
    for b in range(B):
        part_b = partial[b]    # [M,3]
        pred_b = predicted[b]  # [N,3]
        # compute dist => [N,M]
        dist = torch.cdist(pred_b, part_b, p=2) 
        # topk => smallest
        knn_idx = dist.topk(k, largest=False, dim=1).indices  # => [N,k]
        # gather
        neigh_b = []
        for i in range(N):
            row_idx = knn_idx[i]  # => [k]
            neigh_b.append(part_b[row_idx]) # => [k,3]
        neigh_b = torch.stack(neigh_b, dim=0)  # => [N,k,3]
        out.append(neigh_b.unsqueeze(0))
    return torch.cat(out, dim=0)  # => [B,N,k,3]

import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalRefineStage(nn.Module):
    """
    A local refiner that uses a 1D transposed conv to generate expansions 
    for each parent point's local geometry.
    N -> N*r expansions.
    """
    def __init__(self, 
                 in_dim=6,       # (point coords + aggregator = 3 + 3)
                 hidden_dim=128, 
                 out_width=2,    # *2 => doubling
                 k=8):
        super().__init__()
        self.k = k
        self.out_width = out_width

        # aggregator MLP => produce a 'seed feature'
        # We'll define init_width=1 for the 1D conv
        # so final shape => out_width=2
        self.seed_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            # produce hidden_dim*1 => 1D feature map
            nn.Linear(hidden_dim, hidden_dim*1)
        )

        # 1D conv pipeline: from width=1 => out_width=2
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim, 3, kernel_size=1)
        )

    def forward(self, partial, predicted):
        """
        partial:   [B,M,3]
        predicted: [B,N,3]
        => returns [B, N*2, 3]
        """
        B, N, _ = predicted.shape

        # 1) local aggregator => knn => mean
        neighs = local_knn(partial, predicted, k=self.k)  # => [B,N,k,3]
        neigh_mean = neighs.mean(dim=2)                  # => [B,N,3]

        # 2) combine => shape [B,N,6]
        combined = torch.cat([predicted, neigh_mean], dim=-1)  # => [B,N,6]
        combined_2d = combined.view(B*N, 6)                    # => [B*N,6]

        # 3) seed_fc => [B*N, hidden_dim], treat as width=1
        seed_feat = self.seed_fc(combined_2d)  # => [B*N, hidden_dim]
        hidden_dim = seed_feat.shape[1]
        # reshape => [B*N, hidden_dim, 1]
        seed_feat_3d = seed_feat.view(B*N, hidden_dim, 1)

        # 4) conv transpose => from width=1 => 2
        out_map = self.deconv(seed_feat_3d)  # => [B*N, 3, 2]
        # interpret => [B*N,2,3] => [B,N,2,3]
        out_map = out_map.permute(0,2,1).contiguous()  # => [B*N,2,3]
        out_map_4d = out_map.view(B, N, self.out_width, 3)

        # 5) offset each child => [B,N,2,3]
        # parent => [B,N,1,3]
        parent_4d = predicted.view(B,N,1,3)
        child_pts = parent_4d + out_map_4d  # => shape [B,N,2,3]

        # flatten => [B, N*2, 3]
        final_pts = child_pts.view(B, N*self.out_width, 3)
        return final_pts



class Snowflake3StageDecoder(nn.Module):
    """
    Stage1: from ~512 -> e.g. 1024
    Stage2: from 1024 -> 2048
    Stage3: from 2048 -> 4096
    Each stage references partial, does local KNN, expansions
    We'll start from a coarse latent as well for initial points
    """
    def __init__(self, 
                 coarse_num=512,
                 stage1_exp=2, 
                 stage2_exp=2,
                 stage3_exp=2):
        super().__init__()
        self.coarse_num = coarse_num
        self.init_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*self.coarse_num)
        )
        self.stage1 = LocalRefineStage(in_dim=6, hidden_dim=128, out_width=2, k=8)
        self.stage2 = LocalRefineStage(in_dim=6, hidden_dim=128, out_width=2, k=8)
        self.stage3 = LocalRefineStage(in_dim=6, hidden_dim=128, out_width=2, k=8)


    def forward(self, partial_coords, partial_tokens):
        """
        partial_coords: [B,M,3] the partial pointcloud
        partial_tokens: [B,M,d] the transformer output for partial tokens
        => final ~ 4096 points
        """
        B, M, d = partial_tokens.shape
        global_code = partial_tokens.mean(dim=1)  # => [B,d]
        # init coarse
        coarse_1d = self.init_fc(global_code)   # => [B, 3*coarse_num]
        coarse_pts = coarse_1d.view(B, self.coarse_num, 3)

        # stage1 => local refine => ~coarse_num * stage1_exp
        s1_out = self.stage1(partial_coords, coarse_pts)  # => [B,coarse_num * stage1_exp, 3]
        s2_out = self.stage2(partial_coords, s1_out)      # => ...
        s3_out = self.stage3(partial_coords, s2_out)
        return s3_out  # e.g. ~ 4096 if you do 512->1k->2k->4k expansions


class FullModelSnowflake(nn.Module):
    """
    partial_coords => GraphEncoder => multi-token => 
    => geometry transformer => refined tokens => 
    => Snowflake3Stage local refine => final ~4k
    """
    def __init__(self, 
                 in_dim=3,
                 g_hidden_dims=[64,128],
                 g_out_dim=128,
                 t_d_model=128,
                 t_nhead=8,
                 t_layers=4,
                 coarse_num=512,
                 stage1_exp=2,
                 stage2_exp=2,
                 stage3_exp=2):
        super().__init__()
        # 1) graph-based encoder
        self.graph_enc = GraphEncoder(in_dim=in_dim, hidden_dims=g_hidden_dims, out_dim=g_out_dim)

        # 2) geometry-aware transformer
        self.transformer = GeomMultiTokenTransformer(d_model=t_d_model, nhead=t_nhead, num_layers=t_layers)

        # 3) snowflake 3-stage local refine
        self.decoder = Snowflake3StageDecoder(coarse_num=coarse_num,
                                              stage1_exp=stage1_exp,
                                              stage2_exp=stage2_exp,
                                              stage3_exp=stage3_exp)

        # if g_out_dim != t_d_model, we do bridging
        self.bridge_in = nn.Linear(g_out_dim, t_d_model) if g_out_dim!=t_d_model else nn.Identity()

    def forward(self, partial_coords):
        """
        partial_coords: [B,N,3], zeroed for missing
        returns final_points: ~4k
        """
        # graph encode => tokens => [B,N,g_out_dim]
        token_feats = self.graph_enc(partial_coords)  # => [B,N,g_out_dim]

        # bridge => [B,N,t_d_model]
        x = self.bridge_in(token_feats)

        # transform => [B,N,t_d_model]
        x_out = self.transformer(x, partial_coords)

        # decode => local refine => [B,4k,3]
        # we pass partial_coords + x_out to the snowflake decoder
        # but note that snowflake decoder wants partial_coords + partial_tokens
        # partial_tokens ~ the same dimension? We'll assume x_out is the token feats
        final_points = self.decoder(partial_coords, x_out)
        return final_points


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
    #completed: [M, 3]  PyTorch tensor (the networkÅfs output)
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

def train_s3dis_model():
    train_dataset = S3DISDataset(root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis", mask_ratio=0.5, num_points=4096, split='train')
    val_dataset   = S3DISDataset(root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis", mask_ratio=0.5, num_points=4096, split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModelSnowflake(
        in_dim=3,
        g_hidden_dims=[64,128],
        g_out_dim=128,
        t_d_model=128,
        t_nhead=8,
        t_layers=4,
        coarse_num=512,
        stage1_exp=2,
        stage2_exp=2,
        stage3_exp=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss=0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            partial = batch["partial"].to(device) # [B,4096,3]
            full    = batch["full"].to(device)

            optimizer.zero_grad()
            completed = model(partial)  # => ~ [B,4096,3]

            # compute EMD
            loss_emd = emd_loss_sinkhorn(completed, full, reg=0.1, max_iters=50)
            # chamfer
            cd_val, _ = chamfer_distance(completed, full)
            # repulsion
            rep_val = repulsion_loss(completed, k=4, threshold=0.02)
            loss = loss_emd + cd_val + 0.1*rep_val

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}/{num_epochs} => Train Loss: {epoch_loss/len(train_loader):.4f}")

        # minimal val
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_val in val_loader:
                pval = batch_val["partial"].to(device)
                fval = batch_val["full"].to(device)
                cval = model(pval)
                cdv, _ = chamfer_distance(cval, fval)
                val_loss += cdv.item()
        val_loss /= len(val_loader)
        print(f"   Validation (Chamfer): {val_loss:.4f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")

        # Example visualization
        partial_0 = partial[0]
        completed_0 = completed[0]
        original_0 = full[0]
        save_point_cloud_comparison(partial_0, completed_0, original_0, epoch+1)

    return model


if __name__ == "__main__":
    train_s3dis_model()