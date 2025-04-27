import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
import re
from emdloss import emd_loss_sinkhorn

def robust_loadtxt(file_path):
    """
    Reads a .txt file line by line, removing or skipping problematic lines,
    and returns a numpy array of shape [N, 6].
    """
    valid_rows = []
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            # Optionally remove non-ASCII characters
            # line = re.sub(r'[^\x00-\x7F]+','', line)

            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                row_floats = [float(x) for x in parts[:6]]
                valid_rows.append(row_floats)
            except ValueError:
                continue

    return np.array(valid_rows)

def downsample_coords(coords, ratio=0.25):
    """
    coords: [B, N, 3]
    returns: [B, M, 3], M = int(N * ratio)
    """
    B, N, _ = coords.shape
    M = max(1, int(N * ratio))
    idx = torch.randperm(N, device=coords.device)[:M]
    coords_sub = coords[:, idx, :]
    return coords_sub



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

        # Gather .txt files
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
        arr = robust_loadtxt(file_path)  # shape: [N, 6]

        if arr.shape[0] == 0:
            arr = np.zeros((self.num_points, 6), dtype=np.float32)

        N = arr.shape[0]
        if N < self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
        else:
            indices = np.random.choice(N, self.num_points, replace=False)

        sample = arr[indices]  # [num_points, 6]
        coords = sample[:, :3]  # [num_points, 3]

        # ---------------------------------------------------
        # 1) Normalize coords to [-1,1] bounding box (as before)
        # ---------------------------------------------------
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c) / 2.0
        scale = (max_c - min_c).max() / 2.0
        if scale < 1e-8:
            scale = 1.0
        coords = (coords - center) / scale

        # ---------------------------------------------------
        # 2) Masking with a "missing cube" of points
        # ---------------------------------------------------
        # We define a random sub-cube within [-1,1]^3 of side length
        # determined by mask_ratio, or you can pick a fixed fraction.
        side_length = self.mask_ratio  # e.g. 0.5 => a sub-cube of edge 0.5
        # We pick a random center for that sub-cube
        # so that the sub-cube is fully inside [-1,1].
        # half_side = side_length / 2
        half_side = side_length / 2.0

        # random center in [-1+half_side, 1-half_side]
        c = np.random.uniform(-1 + half_side, 1 - half_side, size=3)

        # Then any point within that cube gets zeroed out
        # Condition: |x - cx| <= half_side, etc.
        inside_mask = (
            (coords[:,0] >= c[0] - half_side) & (coords[:,0] <= c[0] + half_side) &
            (coords[:,1] >= c[1] - half_side) & (coords[:,1] <= c[1] + half_side) &
            (coords[:,2] >= c[2] - half_side) & (coords[:,2] <= c[2] + half_side)
        )
        partial_coords = coords.copy()
        partial_coords[inside_mask] = 0.0

        partial_t = torch.from_numpy(partial_coords).float()
        full_t = torch.from_numpy(coords).float()

        return {
            "partial": partial_t,
            "full": full_t
        }





import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance

# If you need KNN or point grouping utilities, you might use libraries like:
# from torch_geometric.nn import knn
# or from open3d.ml import ...
# For demonstration, we'll keep a minimal placeholder.

##################################################################################
# References:
# [1] "PCN: Point Completion Network," X. Yuan et al. (2018)
# [2] "SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer," X. Xiang et al. (2021)
# [3] "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers," M. Yu et al. (NeurIPS 2021)
##################################################################################


# -------------------------------------------------------------------------
# A. Positional Encoding (as before, but let's enlarge the number of frequencies)
# -------------------------------------------------------------------------
class PositionalEncoding3D(nn.Module):
    def __init__(self, num_freqs=6):
        """
        Using more frequencies (e.g., 6) can help the network
        capture finer details of the input geometry.
        """
        super().__init__()
        self.num_freqs = num_freqs

    def forward(self, coords):
        """
        coords: [B, N, 3]
        returns: [B, N, feats]
        """
        B, N, _ = coords.shape
        enc_feats = [coords]

        freqs = 2.0 ** torch.arange(self.num_freqs, device=coords.device).float()
        for freq in freqs:
            for func in [torch.sin, torch.cos]:
                enc_feats.append(func(coords * freq))

        return torch.cat(enc_feats, dim=-1)  # [B, N, 3 + 3*2*num_freqs]


# -------------------------------------------------------------------------
# B. MLP for initial partial point embedding
#    We expand the hidden layer sizes for more capacity.
# -------------------------------------------------------------------------
class PartialPointEmbedding(nn.Module):
    def __init__(self, in_dim=3, embed_dim=64, num_freqs=6):
        super().__init__()
        self.pos_enc = PositionalEncoding3D(num_freqs=num_freqs)
        # If you have color, set in_dim=6, etc.
        # We'll do a bigger MLP: 128 -> 256 -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 2 * in_dim * num_freqs, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, partial_coords):
        """
        partial_coords: [B, N, 3]
        => [B, N, embed_dim]
        """
        x = self.pos_enc(partial_coords)
        return self.mlp(x)


# -------------------------------------------------------------------------
# C. Multi-Scale Feature Extraction (inspired by PCN / Snowflake)
#    We'll do a simple approach: gather features at multiple
#    "resolutions" by random or KNN-based downsampling.
# -------------------------------------------------------------------------
def downsample_random(x, ratio=0.25):
    """
    x: [B, N, d]
    returns: downsampled x at ratio
    """
    B, N, d = x.shape
    M = max(1, int(N * ratio))
    # random subset
    idx = torch.randperm(N, device=x.device)[:M]
    out = x[:, idx, :]  # [B, M, d]
    return out

class MultiScaleExtractor(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_embed, coords):
        """
        x_embed: [B, N, embed_dim]
        coords:  [B, N, 3]
        returns: f1, coords_1, f2, coords_2
        """
        # Scale 1: no further downsampling
        f1 = self.block1(x_embed)
        coords_1 = coords  # same shape as x_embed

        # Scale 2: half
        x_half = downsample_random(x_embed, ratio=0.5)
        coords_half = downsample_coords(coords, ratio=0.5)
        f2 = self.block2(x_half)
        coords_2 = coords_half

        return f1, coords_1, f2, coords_2



# -------------------------------------------------------------------------
# D. "Sparse" Encoder with multi-scale
#    We combine f1 and f2, pass them into a bigger MLP stack
# -------------------------------------------------------------------------
class Naive3DConvEncoder(nn.Module):
    def __init__(self, grid_size=32, in_channels=1, base_channels=8):
        super().__init__()
        self.grid_size = grid_size
        # We treat each voxel as binary occupied or not (or a density).
        # 3D conv layers
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        # Then a small linear to get a latent dimension
        self.fc = nn.Linear((base_channels*4)* (grid_size//8)**3, 128)

    def forward(self, partial_coords):
        """
        partial_coords: [B, N, 3] in [-1,1], with 0 indicating missing points
        We'll voxelize to [B, 1, grid_size, grid_size, grid_size].
        """
        B, N, _ = partial_coords.shape
        device = partial_coords.device
        grid = torch.zeros((B, 1, self.grid_size, self.grid_size, self.grid_size),
                           device=device)

        # naive voxelization
        # scale coords from [-1,1] -> [0, grid_size-1]
        coords_scaled = (partial_coords + 1)/2 * (self.grid_size - 1)
        coords_int = coords_scaled.long().clamp(0, self.grid_size-1)

        for b in range(B):
            for n in range(N):
                x, y, z = coords_int[b,n]
                # Mark voxel as occupied (could also accumulate a density)
                grid[b, 0, z, y, x] = 1.0

        # pass through 3D conv
        feat = self.conv(grid)  
        # flatten
        feat = feat.view(B, -1)  # [B, (base_channels*4)*(grid//8)^3]
        latent = self.fc(feat)   # [B, 128]
        return latent




# -------------------------------------------------------------------------
# E. Geometry-Aware Transformer
#    We'll incorporate a geometric bias as in PoinTr [3].
#    G_ij = alpha * <x_i, x_j>, where x_i are the coords or partial embedding?
#    We'll do a simplified version that just uses the raw coords
#    or the first-scale embedding as a proxy for coords.
# -------------------------------------------------------------------------
import torch
import torch.nn.functional as F

class GeometryAwareAttention(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.dk_per_head = d_model // nhead
        
        # For multi-head: linear layers to project x into Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, coords):
        """
        x: [B, M, d_model]
        coords: [B, M, d_coord]
        Return: [B, M, d_model]
        """
        B, M, d_model = x.shape
        
        # 1) Compute Q, K, V for multi-head
        #    shape => [B, M, d_model], then reshape => [B, nhead, M, d_k]
        Q = self.w_q(x).view(B, M, self.nhead, self.dk_per_head)
        K = self.w_k(x).view(B, M, self.nhead, self.dk_per_head)
        V = self.w_v(x).view(B, M, self.nhead, self.dk_per_head)
        
        # permute => [B, nhead, M, d_k]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # 2) Dot product for attention scores: [B, nhead, M, M]
        #    QK^T / sqrt(d_k)
        #    Q shape: [B, nhead, M, d_k]
        #    K transpose over last two dims => [B, nhead, d_k, M]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk_per_head ** 0.5)
        # => [B, nhead, M, M]
        
        # 3) Compute geometric bias G: [B, M, M], then expand to [B, nhead, M, M]
        G = self.compute_geometric_bias(coords)  # => [B, M, M]
        G = G.unsqueeze(1)                       # => [B, 1, M, M]
        G = G.expand(-1, self.nhead, -1, -1)     # => [B, nhead, M, M]
        
        scores = scores + G  # incorporate geometry

        # 4) Softmax + matmul with V
        attn_weights = F.softmax(scores, dim=-1)          # [B, nhead, M, M]
        out = torch.matmul(attn_weights, V)               # [B, nhead, M, d_k]

        # 5) Merge heads
        out = out.permute(0, 2, 1, 3).contiguous()        # => [B, M, nhead, d_k]
        out = out.view(B, M, self.nhead * self.dk_per_head)  # => [B, M, d_model]

        return out

    def compute_geometric_bias(self, coords):
        """
        coords: [B, M, d_coord]
        Return: G: [B, M, M], where G[b,i,j] = alpha * <coords[b,i], coords[b,j]>
        """
        B, M, d_c = coords.shape
        
        # Flatten for matmul
        coords_2d = coords.view(B*M, d_c)  # [B*M, d_c]
        dot_matrix = torch.matmul(coords_2d, coords_2d.T) # [B*M, B*M]

        # block diagonal extract
        G = torch.zeros((B, M, M), device=coords.device)
        for b in range(B):
            start = b*M
            end   = (b+1)*M
            block = dot_matrix[start:end, start:end]  # [M, M]
            G[b] = block

        G = self.alpha * G
        return G


class GeometricAwareTransformer(nn.Module):
    """
    We'll do multiple layers of GeometryAwareAttention + FFN.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                GeometryAwareAttention(d_model, nhead),
                nn.Sequential(
                    nn.Linear(d_model, d_model*2),
                    nn.ReLU(True),
                    nn.Linear(d_model*2, d_model),
                )
            ]))
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, coords):
        """
        x: [B, M, d_model]
        coords: [B, M, d_coord]
        """
        for i, (attn, ffn) in enumerate(self.layers):
            # 1) Self-Attention with geometry bias
            x_attn = attn(x, coords)
            x = x + x_attn
            x = self.norms1[i](x)

            # 2) FFN
            x_ffn = ffn(x)
            x = x + x_ffn
            x = self.norms2[i](x)
        return x


# -------------------------------------------------------------------------
# F. Hierarchical Decoder
#    Now more robust with multiple refinement steps and residual MLP blocks
# -------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class Deconv1DDecoder(nn.Module):
    """
    A 1D transposed convolution decoder to generate M points in 3D.
    We treat the 'width' dimension as the "number of points."
    Channels -> intermediate features. We'll end with 3 channels for (x,y,z).
    """
    def __init__(self, 
                 latent_dim=128,
                 hidden_c=256,       # base channel size
                 out_points=4096     # final # of points
                ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_points = out_points

        # We'll pick a small initial 'width' (1 or 2) to start from.
        # Then a series of ConvTranspose1d layers to reach out_points.

        # 1) We'll embed latent into a [B, hidden_c, init_width]
        self.init_width = 8  # e.g., 8 => we will upsample from 8 -> out_points
        self.init_fc = nn.Linear(latent_dim, hidden_c * self.init_width)

        # 2) Deconvolution stack to upsample from width=8 -> out_points
        #    We'll do it in a few steps. The exact strides/paddings
        #    might need tweaking to match exactly out_points.
        #    For simplicity, let's do multi-step or a single step if out_points isn't huge.

        # Example: if we want 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024
        # we might do repeated conv transpose with stride=2.
        # But that can be many layers. Let's do a small version.
        self.deconv_blocks = nn.Sequential(
            # Deconv1: from width=8 => 16
            nn.ConvTranspose1d(hidden_c, hidden_c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # Deconv2: 16 => 32
            nn.ConvTranspose1d(hidden_c, hidden_c//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # Deconv3: 32 => 64
            nn.ConvTranspose1d(hidden_c//2, hidden_c//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # ...
            # we can keep going until we exceed out_points, then do a final conv to get EXACT out_points
            # or we do a final interpolation.
        )

        # 3) Final conv: reduce channels -> 3, and match exact out_points
        # We'll do that in forward() by interpolation if needed.
        self.to_xyz = nn.Conv1d(hidden_c//4, 3, kernel_size=1)

    def forward(self, latent):
        """
        latent: [B, latent_dim]
        returns: coords: [B, out_points, 3]
        """
        B, D = latent.shape
        # step (a): map latent -> [B, hidden_c, init_width]
        feat_init = self.init_fc(latent)  # [B, hidden_c*init_width]
        # reshape => [B, hidden_c, init_width]
        feat_init = feat_init.view(B, -1, self.init_width)

        # step (b): pass through deconv stack => expand in 1D dimension
        feat = self.deconv_blocks(feat_init)  # shape e.g. [B, c, w]

        # if final w < out_points, we can further interpolate
        # if final w > out_points, we can slice or do a final conv with stride
        cur_w = feat.shape[2]
        if cur_w < self.out_points:
            # 1D interpolation up to out_points
            feat = F.interpolate(feat, size=self.out_points, mode='linear', align_corners=False)
        elif cur_w > self.out_points:
            feat = feat[:, :, :self.out_points]

        # step (c): final conv => [B, 3, out_points]
        xyz_map = self.to_xyz(feat)  # => [B, 3, out_points]

        # step (d): permute => [B, out_points, 3]
        coords = xyz_map.permute(0, 2, 1).contiguous()

        # shape check
        # coords shape => [B, out_points, 3]
        return coords

import torch
import torch.nn as nn
import torch.nn.functional as F

class Deconv2DDecoder(nn.Module):
    """
    A 2D transposed convolution 'decoder' that outputs a grid of points [H, W] in 3D.

    Steps:
      1) Map latent -> small 2D feature (C, H_0, W_0).
      2) Upsample via ConvTranspose2d to final (C, H, W).
      3) Last conv => 3 channels => interpret as (x, y, z).
      4) Flatten => [B, H*W, 3] point cloud.
    """
    def __init__(self, 
                 latent_dim=128,
                 init_channels=256,
                 init_spatial=(4,4),     # starting size H0,W0
                 final_spatial=(32,32), # final size H,W
                 out_channels=3):       # we want (x,y,z)
        super().__init__()
        self.latent_dim = latent_dim
        self.init_channels = init_channels
        self.init_spatial = init_spatial
        self.final_spatial = final_spatial
        self.out_channels = out_channels

        H0, W0 = init_spatial
        # 1) Project latent -> [B, init_channels, H0, W0]
        self.init_fc = nn.Linear(latent_dim, init_channels * H0 * W0)

        # We'll define a small transposed conv stack. The exact strides,
        # kernel sizes, etc. might need tweaking to reach final_spatial exactly.
        # We'll do an example with stride=2 blocks to roughly double each dimension.

        # 2) A sequence of upsampling conv layers
        self.deconv_stack = nn.Sequential(
            # e.g. from (C,4,4)->(C,8,8)
            nn.ConvTranspose2d(init_channels, init_channels//2,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(init_channels//2, init_channels//4,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # We can keep adding or adjust the number of layers to approach final_spatial
        )

        # 3) Final conv => out_channels (3) + upsample if needed
        self.to_xyz = nn.Conv2d(init_channels//4, self.out_channels, kernel_size=1)

    def forward(self, latent):
        """
        latent: [B, latent_dim]
        returns: coords [B, H*W, 3]
        """
        B, D = latent.shape
        H0, W0 = self.init_spatial
        # (a) Map latent -> [B, init_channels, H0, W0]
        feat_init = self.init_fc(latent)  # => [B, C*H0*W0]
        feat_init = feat_init.view(B, self.init_channels, H0, W0)

        # (b) Deconvolve/upsample
        feat = self.deconv_stack(feat_init)  # => [B, c, Hx, Wx]

        # If final (Hx, Wx) < or > final_spatial, we do an interpolation or slice.
        final_H, final_W = self.final_spatial
        cur_H, cur_W = feat.shape[2], feat.shape[3]

        # Adjust if needed
        if (cur_H, cur_W) != (final_H, final_W):
            feat = F.interpolate(feat, size=(final_H, final_W), mode='bilinear', align_corners=False)

        # (c) to_xyz => [B, 3, final_H, final_W]
        xyz_map = self.to_xyz(feat)

        # (d) flatten => [B, 3, final_H*final_W]
        # then permute => [B, final_H*final_W, 3]
        Bc, Cc, Hc, Wc = xyz_map.shape
        xyz_map_2d = xyz_map.view(Bc, Cc, -1)    # [B, 3, H*W]
        coords = xyz_map_2d.permute(0, 2, 1)     # [B, H*W, 3]

        return coords

import torch
import torch.nn as nn
import torch.nn.functional as F

class Patch2DRefiner(nn.Module):
    """
    Produces a single patch of points:
      1) Start from a small 2D feature (4,4)
      2) Upsample to (8,8), then (16,16)
      3) Final conv => 3 channels (XYZ)
      4) Flatten => [B, 256, 3]
    """
    def __init__(self, 
                 patch_latent_dim=128,
                 base_channels=128,     # base # of channels for conv
                 init_size=(4,4),      # start resolution
                 final_size=(16,16)    # end resolution
                ):
        super().__init__()
        self.patch_latent_dim = patch_latent_dim
        self.base_channels = base_channels
        self.init_size = init_size
        self.final_size = final_size

        H0, W0 = init_size
        # 1) Map patch_latent -> [B, base_channels, 4,4]
        self.init_fc = nn.Linear(patch_latent_dim, base_channels * H0 * W0)

        # 2) A small upsample stack
        #   - upsample from (4,4)->(8,8)->(16,16) in two stages
        #   - each stage: ConvTranspose2d(..., stride=2)
        self.upsample_stack = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels//2,
                               kernel_size=4, stride=2, padding=1),  # (4,4)->(8,8)
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels//2, base_channels//4,
                               kernel_size=4, stride=2, padding=1),  # (8,8)->(16,16)
            nn.ReLU(True),
        )

        # 3) Final conv => 3 channels
        self.to_xyz = nn.Conv2d(base_channels//4, 3, kernel_size=1)

    def forward(self, patch_latent):
        """
        patch_latent: [B, patch_latent_dim]
        returns: [B, (16*16), 3] = [B, 256, 3]
        """
        B, D = patch_latent.shape
        H0, W0 = self.init_size
        HF, WF = self.final_size

        # (a) Map latent -> [B, base_channels, 4,4]
        feat_init = self.init_fc(patch_latent)  # => [B, C*4*4]
        feat_init = feat_init.view(B, self.base_channels, H0, W0)

        # (b) Upsample stack => e.g. (4,4)->(8,8)->(16,16)
        feat = self.upsample_stack(feat_init)  # => [B, c, 16,16]

        # (c) final conv => 3 channels
        xyz_map = self.to_xyz(feat)  # => [B, 3, 16,16]

        # (d) flatten => [B, 3, 256] => permute => [B, 256, 3]
        Bc, Cc, Hc, Wc = xyz_map.shape  # e.g. (B, 3, 16,16)
        xyz_map_2d = xyz_map.view(Bc, Cc, -1)     # => [B, 3, 256]
        coords = xyz_map_2d.permute(0, 2, 1)      # => [B, 256, 3]

        return coords

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPatch2DRefiner(nn.Module):
    """
    A deeper patch-based 2D transposed-conv refiner.
    By default: (4->8->16->32->64) final => (64*64)=4096 points from ONE patch.
    If you want multiple patches (e.g., 16 patches * 256 each), reduce final_size etc.
    """
    def __init__(self, 
                 patch_latent_dim=128,
                 base_channels=256,     # bigger channels
                 init_size=(4,4),
                 final_size=(64,64)     # 64x64 => 4096 points
                ):
        super().__init__()
        self.patch_latent_dim = patch_latent_dim
        self.base_channels = base_channels
        self.init_size = init_size
        self.final_size = final_size

        H0, W0 = init_size
        # 1) Map patch_latent -> [B, base_channels, 4,4]
        self.init_fc = nn.Linear(patch_latent_dim, base_channels * H0 * W0)

        # 2) "Double the depth" vs older version: more upsampling blocks
        #    (4->8)->(8->16)->(16->32)->(32->64) => 4 transposed conv layers
        ch = base_channels
        self.upsample_stack = nn.Sequential(
            nn.ConvTranspose2d(ch, ch//2, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ch//2, ch//4, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(True),

            nn.ConvTranspose2d(ch//4, ch//4, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(True),

            nn.ConvTranspose2d(ch//4, ch//8 if (ch//8)>=8 else 8,  # clamp to at least 8 channels
                               kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(True),
        )

        # 3) Final conv => 3 channels
        #    We assume now the result is 64x64 => 4096 points
        #    If you want fewer final points, reduce final_size or reduce # of blocks
        final_ch = max(ch//8, 8)  # ensure at least 8 channels remain
        self.to_xyz = nn.Conv2d(final_ch, 3, kernel_size=1)

    def forward(self, patch_latent):
        """
        patch_latent: [B, patch_latent_dim]
        returns: [B, (final_H*final_W), 3]
        e.g. final=(64,64) => 4096 points.
        """
        B, D = patch_latent.shape
        H0, W0 = self.init_size
        HF, WF = self.final_size  # e.g. (64,64)

        # (a) map latent -> [B, base_channels, H0, W0]
        feat_init = self.init_fc(patch_latent)
        feat_init = feat_init.view(B, self.base_channels, H0, W0)

        # (b) upsample
        feat = self.upsample_stack(feat_init)

        # If final size not exactly matched, do an interpolate
        cur_H, cur_W = feat.shape[2], feat.shape[3]
        if (cur_H, cur_W) != (HF, WF):
            feat = F.interpolate(feat, size=(HF, WF), mode='bilinear', align_corners=False)

        # (c) final conv => 3 channels
        xyz_map = self.to_xyz(feat)  # [B, 3, HF, WF]

        # flatten => [B, 3, HF*WF] => permute => [B, HF*WF, 3]
        Bc, Cc, Hc, Wc = xyz_map.shape
        xyz_2d = xyz_map.view(Bc, Cc, -1)  # [B, 3, HF*WF]
        coords = xyz_2d.permute(0, 2, 1).contiguous()  # [B, HF*WF, 3]
        return coords



class DeepPatch2DDecoder(nn.Module):
    """
    Multi-patch approach, each patch is a deeper 2D transposed conv.
    Suppose we want 4 patches, each => (32x32)=1024, total => 4096.
    """
    def __init__(self,
                 global_dim=256,
                 patch_latent_dim=128,
                 num_patches=4,
                 base_channels=256,
                 init_size=(4,4),
                 final_size=(32,32)):  # => 1024 per patch
        super().__init__()
        self.global_dim = global_dim
        self.patch_latent_dim = patch_latent_dim
        self.num_patches = num_patches

        # produce patch latents
        self.patch_latent_fc = nn.Sequential(
            nn.Linear(global_dim, num_patches * patch_latent_dim),
            nn.ReLU(True),
        )

        # each patch is a "DeepPatch2DRefiner" but we set final_size=(32,32)
        self.patch_refiners = nn.ModuleList([
            DeepPatch2DRefiner(
                patch_latent_dim=patch_latent_dim,
                base_channels=base_channels,
                init_size=init_size,
                final_size=final_size
            )
            for _ in range(num_patches)
        ])

    def forward(self, global_latent):
        """
        global_latent: [B, global_dim]
        return: [B, 4096, 3]
        """
        B, D = global_latent.shape
        # 1) map to patch latents
        patch_latent_vec = self.patch_latent_fc(global_latent)
        patch_latent_vec = patch_latent_vec.view(B, self.num_patches, self.patch_latent_dim)

        # 2) for each patch => ~1024 points
        all_patches = []
        for i in range(self.num_patches):
            patch_latent_i = patch_latent_vec[:, i, :]
            coords_i = self.patch_refiners[i](patch_latent_i)  # => [B, 1024, 3]
            all_patches.append(coords_i)

        # 3) concat => [B, 4096, 3]
        completed = torch.cat(all_patches, dim=1)
        return completed






# -------------------------------------------------------------------------
# G. Full Model
# -------------------------------------------------------------------------
class PointCloudCompletionNetwork(nn.Module):
    def __init__(self, grid_size=32, base_channels=8, 
                 latent_dim=256, 
                 coarse_num=512, 
                 refinement_steps=2):
        super().__init__()
        self.encoder = Naive3DConvEncoder(grid_size=grid_size,
                                          in_channels=1,
                                          base_channels=base_channels)
        self.decoder = DeepPatch2DDecoder(
            patch_latent_dim=128,
            num_patches=4,
            base_channels=256,  # bigger
            init_size=(4,4),
            final_size=(32,32)  # 1024 per patch
        )
        # optional bridging if needed:
        self.bridge = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(True),
        )

    def forward(self, partial_coords):
        """
        partial_coords: [B, N, 3]
        returns: [B, M, 3]
        """
        latent = self.encoder(partial_coords)  # e.g. [B, 128]
        latent = self.bridge(latent)           # [B, latent_dim]
        completed = self.decoder(latent)       # [B, coarse_num, 3]
        return completed





import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_point_cloud_comparison(partial, completed, original, epoch, out_dir="visuals"):
    """
    partial:   [N, 3]  PyTorch tensor (the masked, incomplete input)
    completed: [M, 3]  PyTorch tensor (the network’s output)
    original:  [N, 3]  PyTorch tensor (the unmasked, full point cloud)
    Saves an image 'completion_epoch_{epoch}.png' with 3 side-by-side subplots.
    """
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




import os
import time
import torch.optim as optim
from tqdm import tqdm  # <-- progress bar

import torch.nn.functional as F

def pairwise_dist(points):
    """
    points: [B, N, 3]
    returns: [B, N, N] pairwise squared distances
    """
    B, N, _ = points.shape
    xx = (points * points).sum(dim=-1, keepdim=True)  # [B, N, 1]
    dist = xx + xx.transpose(1,2) - 2 * torch.matmul(points, points.transpose(1,2))
    dist = F.relu(dist)  # clamp negative to 0
    return dist

def repulsion_loss(pred_points, k=4, threshold=0.02):
    """
    Penalize pairs of points that are too close (below threshold).
    For each point, we look at its k nearest neighbors.
    """
    B, N, _ = pred_points.shape
    dist_mat = pairwise_dist(pred_points)  # [B, N, N]

    # top-k nearest distances (excluding self => 0)
    knn_vals, _ = torch.topk(dist_mat, k=k+1, dim=-1, largest=False)
    # shape: [B, N, k+1]
    # skip the first neighbor (self-dist=0)
    knn_vals = knn_vals[..., 1:]  # => [B, N, k]

    # repulsion term: if dist < threshold, penalty = threshold - dist
    rep = F.relu(threshold - knn_vals)
    # average
    rep_loss_val = torch.mean(rep)
    return rep_loss_val


def train_s3dis_model():
    # 1) Create dataset & loader
    train_dataset = S3DISDataset(
        root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis",
        mask_ratio=0.5,
        num_points=4096,
        split='train'
    )
    val_dataset = S3DISDataset(
        root=r"E:\S3DIS\cvg-data.inf.ethz.ch\s3dis",
        mask_ratio=0.5,
        num_points=4096,
        split='val'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=4
    )

    # 2) Instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointCloudCompletionNetwork(
        grid_size=32,
        base_channels=8,
        latent_dim=256,
        coarse_num=1024,
        refinement_steps=2
    ).to(device)

    # 3) Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100
    alpha_rep = 0.1  # repulsion weight

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for i, batch in enumerate(train_iter):
            partial = batch["partial"].to(device)  # [B, N, 3]
            full = batch["full"].to(device)        # [B, N, 3]

            optimizer.zero_grad()
            completed = model(partial)             # [B, M, 3]

            #print(completed)

            #cd_loss, _ = chamfer_distance(completed, full)
            loss_emd = emd_loss_sinkhorn(completed, full, reg=0.1, max_iters=50)
            #rep_loss_val = repulsion_loss(completed, k=4, threshold=0.02)
            loss = loss_emd #+ alpha_rep * rep_loss_val

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_iter.set_postfix({"loss": loss.item()})

        epoch_loss_avg = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss_avg:.4f}")

        # -- validation minimal
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_val in val_loader:
                pval = batch_val["partial"].to(device)
                fval = batch_val["full"].to(device)
                cval = model(pval)
                cdl, _ = chamfer_distance(cval, fval)
                val_loss += cdl.item()
        val_loss_avg = val_loss / len(val_loader)
        print(f"   Validation Loss: {val_loss_avg:.4f}")

        # 6) Save model checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")

        # 7) Save an example image (just the first batch from training loop)
        #    We'll reuse the last 'partial' & 'completed' from the train loop.
        #    We'll pick the first item in that batch [B=4].
        partial_0   = partial[0]      # [N, 3] from the batch
        completed_0 = completed[0]    # [M, 3] from the model
        original_0  = full[0]         # [N, 3] unmasked

        save_point_cloud_comparison(partial_0, completed_0, original_0, epoch+1)


    print("Training complete!")
    return model


if __name__ == "__main__":
    train_s3dis_model()