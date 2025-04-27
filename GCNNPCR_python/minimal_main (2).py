import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from pytorch3d.loss import chamfer_distance
# We'll assume you have a local file emdloss.py with `emd_loss_sinkhorn`.
from emdloss import emd_loss_sinkhorn

##############################################################################
# 1) Dataset (same or similar to your snippet)
##############################################################################

class S3DISDataset(Dataset):
    def __init__(self, 
                 root: str,
                 mask_ratio: float = 0.5,  # fraction of points to remove
                 num_points: int = 4096,
                 split: str = 'train'):
        """
        Args:
            mask_ratio: fraction of the point cloud (0.0~1.0) to randomize as 'missing'
        """
        super().__init__()
        self.root = root
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.split = split

        # Gather .txt files
        pattern = os.path.join(root, '**', '*.txt')
        all_files = [f for f in glob(pattern, recursive=True)
                     if ('alignmentAngle' not in f and 'Annotations' not in f)]

        # Simple 80/20 split
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
            # fallback if file is empty or invalid
            arr = np.zeros((self.num_points, 6), dtype=np.float32)

        # 1) sample num_points from arr
        N = arr.shape[0]
        if N < self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
        else:
            indices = np.random.choice(N, self.num_points, replace=False)

        sample = arr[indices]  # [num_points, 6]
        coords = sample[:, :3]  # [num_points, 3]

        # 2) Normalize coords to [-1,1]
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c) / 2.0
        scale = (max_c - min_c).max() / 2.0
        if scale < 1e-8:
            scale = 1.0
        coords = (coords - center) / scale

        # 3) Randomly remove 'mask_ratio' fraction of points
        #    We'll define mask_size => number of points to "remove"
        mask_size = int(self.num_points * self.mask_ratio)
        mask_indices = np.random.choice(self.num_points, mask_size, replace=False)

        partial_coords = coords.copy()
        # Set those randomly chosen indices to [0,0,0]
        partial_coords[mask_indices] = 0.0

        # 4) Convert to torch
        partial_t = torch.from_numpy(partial_coords).float()  # [num_points, 3]
        full_t    = torch.from_numpy(coords).float()         # [num_points, 3]

        return {
            "partial": partial_t,
            "full": full_t
        }



def robust_loadtxt(file_path):
    """
    Reads .txt line by line, skipping problematic lines, returning [N,6].
    """
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



##############################################################################
# 3) ADVANCED ARCHITECTURE (Encoder/Transformer/Decoder)
##############################################################################

# 3.1 Deeper partial embedding / multi-scale approach
class PositionalEncoding3D(nn.Module):
    def __init__(self, num_freqs=6):
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


class PartialPointEmbedding(nn.Module):
    """
    Deeper MLP to embed partial coords -> embed_dim, using positional encoding.
    """
    def __init__(self, in_dim=3, embed_dim=64, num_freqs=6):
        super().__init__()
        self.pos_enc = PositionalEncoding3D(num_freqs=num_freqs)
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
        #simple mean
        #x = x.mean(dim=1)
        return self.mlp(x)


def downsample_random(x, ratio=0.5):
    B, N, d = x.shape
    M = int(N*ratio)
    idx = torch.randperm(N, device=x.device)[:M]
    return x[:, idx, :]


class MultiScaleExtractor(nn.Module):
    """
    Deeper extraction or multi-scale approach to partial embeddings.
    """
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

    def forward(self, x_embed):
        """
        x_embed: [B, N, embed_dim]
        returns two scales f1, f2
        """
        B, N, C = x_embed.shape
        # scale1: entire
        s1_2d = x_embed.view(B*N, C)
        f1_2d = self.block1(s1_2d)
        f1 = f1_2d.view(B, N, -1)

        # scale2: half
        x_half = downsample_random(x_embed, ratio=0.5)
        Bh, Mh, _ = x_half.shape
        xh_2d = x_half.view(Bh*Mh, C)
        f2_2d = self.block2(xh_2d)
        f2 = f2_2d.view(Bh, Mh, -1)
        return f1, f2


class DeepEncoder(nn.Module):
    """
    partial_coords -> [embedding] -> [multi-scale] -> global pooling -> [B, global_dim]
    """
    def __init__(self, in_dim=3, embed_dim=64, global_dim=256):
        super().__init__()
        self.embed = PartialPointEmbedding(in_dim=in_dim, embed_dim=embed_dim)
        self.mscale = MultiScaleExtractor(embed_dim=embed_dim)
        self.final_fc = nn.Sequential(
            nn.Linear(embed_dim*2, 512),
            nn.ReLU(True),
            nn.Linear(512, global_dim)
        )

    def forward(self, partial_coords):
        """
        partial_coords: [B, N, 3]
        returns: [B, global_dim]
        """
        x_embed = self.embed(partial_coords)  # => [B, N, 64]
        f1, f2 = self.mscale(x_embed)         # => [B,N,64], [B,N/2,64]
        g1 = f1.mean(dim=1)                  # => [B,64]
        g2 = f2.mean(dim=1)                  # => [B,64]
        catg = torch.cat([g1, g2], dim=1)    # => [B,128]
        out = self.final_fc(catg)            # => [B, global_dim]
        return out


##############################################################################
# 3.2 Optional geometry-aware Transformer
##############################################################################

class GeometryAwareAttention(nn.Module):
    def __init__(self, d_model=128, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.dk = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, coords):
        """
        x: [B, M, d_model]
        coords: [B, M, 3]
        Return: [B, M, d_model]
        """
        B, M, D = x.shape

        Q = self.w_q(x).view(B, M, self.nhead, self.dk).permute(0,2,1,3)
        K = self.w_k(x).view(B, M, self.nhead, self.dk).permute(0,2,1,3)
        V = self.w_v(x).view(B, M, self.nhead, self.dk).permute(0,2,1,3)

        # Dot product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk**0.5)  # [B,nhead,M,M]

        # geometry bias
        G = self.compute_geom_bias(coords)  # [B,M,M]
        G = G.unsqueeze(1).expand(-1,self.nhead,-1,-1)  # [B,nhead,M,M]
        scores = scores + G

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B,nhead,M,dk]
        out = out.permute(0,2,1,3).contiguous().view(B, M, self.nhead*self.dk)
        return out

    def compute_geom_bias(self, coords):
        B, M, _ = coords.shape
        coords2d = coords.view(B*M, -1)
        dot_matrix = torch.matmul(coords2d, coords2d.t())  # [B*M, B*M]
        G = torch.zeros((B,M,M), device=coords.device)
        for b in range(B):
            st = b*M
            ed = (b+1)*M
            G[b] = dot_matrix[st:ed, st:ed]
        return self.alpha * G


class GeometricAwareTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GeometryAwareAttention(d_model, nhead)
            ff = nn.Sequential(
                nn.Linear(d_model, d_model*2),
                nn.ReLU(True),
                nn.Linear(d_model*2, d_model),
            )
            self.layers.append(nn.ModuleList([attn, ff]))
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, coords):
        """
        x: [B, M, d_model]
        coords: [B, M, 3]
        """
        for i, (attn, ff) in enumerate(self.layers):
            x_attn = attn(x, coords)
            x = self.norm1[i](x + x_attn)
            x_ff = ff(x)
            x = self.norm2[i](x + x_ff)
        return x


##############################################################################
# 3.3 Multi-Stage Patch-based Decoder (Snowflake-style)
##############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class Patch2DRefinerWithSkip(nn.Module):
    """
    Combines patch_latent + partial_embed => final 2D upsample
    Yields (finalH x finalW) points
    """
    def __init__(self, 
                 patch_latent_dim=128,
                 partial_dim=64,
                 base_channels=128,
                 init_size=(4,4),
                 final_size=(16,16)):
        super().__init__()
        self.patch_latent_dim = patch_latent_dim
        self.partial_dim = partial_dim
        self.base_channels = base_channels
        self.init_size = init_size
        self.final_size = final_size

        H0, W0 = init_size
        HF, WF = final_size

        # Merge (patch_latent + partial_embed) => feed in
        self.fc_in = nn.Linear(patch_latent_dim + partial_dim, base_channels * H0 * W0)

        # Example upsample stack (2 steps)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels//2, base_channels//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.to_xyz = nn.Conv2d(base_channels//4, 3, kernel_size=1)

    def forward(self, patch_latent, partial_embed):
        """
        patch_latent: [B, patch_latent_dim]
        partial_embed: [B, N, partial_dim] -> [B, partial_dim]
        returns: coords => [B, final_size[0]*final_size[1], 3]
        """
        B, _ = patch_latent.shape
        #simple mean
        partial_embed = partial_embed.mean(dim=1)
        combined = torch.cat([patch_latent, partial_embed], dim=1)  # [B, patch_latent_dim+partial_dim]
        H0, W0 = self.init_size
        HF, WF = self.final_size

        feat_init = self.fc_in(combined).view(B, self.base_channels, H0, W0)
        feat = self.deconv(feat_init)
        curH, curW = feat.shape[2], feat.shape[3]
        if (curH, curW) != (HF, WF):
            feat = F.interpolate(feat, size=(HF, WF), mode='bilinear', align_corners=False)
        xyz_map = self.to_xyz(feat)  # => [B,3,HF,WF]
        Bc, Cc, Hc, Wc = xyz_map.shape
        out2d = xyz_map.view(Bc, Cc, -1)
        coords = out2d.permute(0,2,1)
        return coords


class Patch2DRefiner(nn.Module):
    """
    Produce (finalH * finalW) points from a patch latent.
    For example, (16x16)=256 or (32x32)=1024 etc.
    """
    def __init__(self, patch_latent_dim=128, base_channels=128,
                 init_size=(4,4), final_size=(16,16)):
        super().__init__()
        self.patch_latent_dim = patch_latent_dim
        self.base_channels = base_channels
        self.init_size = init_size
        self.final_size = final_size

        H0, W0 = init_size
        self.init_fc = nn.Linear(patch_latent_dim, base_channels * H0 * W0)

        # a small transposed conv stack
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels//2, base_channels//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.to_xyz = nn.Conv2d(base_channels//4, 3, kernel_size=1)

    def forward(self, latent):
        B, D = latent.shape
        H0, W0 = self.init_size
        HF, WF = self.final_size

        feat_init = self.init_fc(latent).view(B, self.base_channels, H0, W0)
        feat = self.deconv(feat_init)
        if (feat.shape[2], feat.shape[3]) != (HF, WF):
            feat = F.interpolate(feat, size=(HF,WF), mode='bilinear', align_corners=False)
        xyz = self.to_xyz(feat)  # => [B,3,HF,WF]
        Bc, Cc, Hc, Wc = xyz.shape
        out2d = xyz.view(Bc, Cc, -1)   # => [B,3, HF*WF]
        coords = out2d.permute(0,2,1)  # => [B, HF*WF, 3]
        return coords

class MultiPatchStageWithSkip(nn.Module):
    def __init__(self,
                 global_dim=256,
                 patch_latent_dim=128,
                 partial_dim=64,
                 num_patches=16,
                 patch_refiner=None):
        super().__init__()
        self.global_dim = global_dim
        self.patch_latent_dim = patch_latent_dim
        self.partial_dim = partial_dim
        self.num_patches = num_patches
        self.patch_fc = nn.Sequential(
            nn.Linear(global_dim, num_patches*patch_latent_dim),
            nn.ReLU(True),
        )
        # We'll replicate the same patch_refiner for each patch
        # Or you can define distinct ones.
        self.patch_refiners = nn.ModuleList([
            patch_refiner for _ in range(num_patches)
        ])

    def forward(self, global_feat, partial_embed):
        """
        global_feat:   [B, global_dim]
        partial_embed: [B, partial_dim]
        => [B, total_points, 3]
        """
        B, D = global_feat.shape
        latents_all = self.patch_fc(global_feat)  # => [B, num_patches*patch_latent_dim]
        latents_all = latents_all.view(B, self.num_patches, self.patch_latent_dim)

        all_coords = []
        for i in range(self.num_patches):
            lat_i = latents_all[:, i, :]
            coords_i = self.patch_refiners[i](lat_i, partial_embed)
            all_coords.append(coords_i)
        coords = torch.cat(all_coords, dim=1)
        return coords


class MultiPatchStage(nn.Module):
    """
    One stage that uses multiple patches => sum up the points.
    """
    def __init__(self, global_dim=256, patch_latent_dim=128, num_patches=16, patch_refiner=None):
        super().__init__()
        self.global_dim = global_dim
        self.patch_latent_dim = patch_latent_dim
        self.num_patches = num_patches
        self.fc = nn.Sequential(
            nn.Linear(global_dim, num_patches*patch_latent_dim),
            nn.ReLU(True),
        )
        # Use the same patch refiner or distinct?
        self.patch_refiner = nn.ModuleList([patch_refiner for _ in range(num_patches)])

    def forward(self, global_feat):
        B, D = global_feat.shape
        patch_latents = self.fc(global_feat)  # => [B, num_patches*patch_latent_dim]
        patch_latents = patch_latents.view(B, self.num_patches, self.patch_latent_dim)

        out_coords = []
        for i in range(self.num_patches):
            lat_i = patch_latents[:, i, :]
            coords_i = self.patch_refiner[i](lat_i)
            out_coords.append(coords_i)
        coords = torch.cat(out_coords, dim=1)  # => [B, total_pts, 3]
        return coords

class MultiStageDecoderWithSkip(nn.Module):
    """
    Now with 3 stages: 
      Stage1 -> e.g. 1024 total
      Stage2 -> e.g. 4096 total
      Stage3 -> e.g. 16384 total
    """
    def __init__(self,
                 global_dim=256,
                 partial_dim=64,
                 # Stage1
                 s1_num_patches=16,
                 s1_patch_latent_dim=128,
                 s1_patch_refiner=None,
                 # Stage2
                 s2_num_patches=16,
                 s2_patch_latent_dim=128,
                 s2_patch_refiner=None,
                 # Stage3
                 s3_num_patches=16,
                 s3_patch_latent_dim=128,
                 s3_patch_refiner=None,
                 #resample
                 final_out_n=4096,):
        super().__init__()
        # If none provided, define defaults
        if s1_patch_refiner is None:
            # produce 64 points/patch => 16 patches => 1024
            s1_patch_refiner = Patch2DRefinerWithSkip(
                patch_latent_dim=s1_patch_latent_dim,
                partial_dim=partial_dim,
                base_channels=128,
                init_size=(4,4),
                final_size=(8,8)
            )
        if s2_patch_refiner is None:
            # produce 256 points/patch => 16 patches => 4096
            s2_patch_refiner = Patch2DRefinerWithSkip(
                patch_latent_dim=s2_patch_latent_dim,
                partial_dim=partial_dim,
                base_channels=256,
                init_size=(8,8),
                final_size=(16,16)
            )
        if s3_patch_refiner is None:
            # produce 1024 points/patch => 16 patches => 16384
            s3_patch_refiner = Patch2DRefinerWithSkip(
                patch_latent_dim=s3_patch_latent_dim,
                partial_dim=partial_dim,
                base_channels=256,
                init_size=(16,16),
                final_size=(32,32)
            )

        self.stage1 = MultiPatchStageWithSkip(
            global_dim=global_dim,
            patch_latent_dim=s1_patch_latent_dim,
            partial_dim=partial_dim,
            num_patches=s1_num_patches,
            patch_refiner=s1_patch_refiner
        )
        self.stage2 = MultiPatchStageWithSkip(
            global_dim=global_dim,
            patch_latent_dim=s2_patch_latent_dim,
            partial_dim=partial_dim,
            num_patches=s2_num_patches,
            patch_refiner=s2_patch_refiner
        )
        self.stage3 = MultiPatchStageWithSkip(
            global_dim=global_dim,
            patch_latent_dim=s3_patch_latent_dim,
            partial_dim=partial_dim,
            num_patches=s3_num_patches,
            patch_refiner=s3_patch_refiner
        )
        self.final_out_n=final_out_n

    def forward(self, global_feat, partial_embed):
        # Stage1 => ~1k
        coords1 = self.stage1(global_feat, partial_embed)
        # optionally aggregator => new global feat or skip
        # We'll keep it simple: re-use same global_feat
        # Stage2 => ~4k
        coords2 = self.stage2(global_feat, partial_embed)
        # Stage3 => ~16k
        coords3 = self.stage3(global_feat, partial_embed)

        #resample coords
        final_coords = random_resample(coords3, out_n=self.final_out_n)
        return final_coords


class MultiStageDecoder(nn.Module):
    """
    E.g. 2-stage approach:
      stage1 => 1k points
      stage2 => 4k points
    """
    def __init__(self, 
                 global_dim=256,
                 # stage1
                 s1_num_patches=16,
                 s1_patch_latent_dim=128,
                 s1_refiner=None,
                 # stage2
                 s2_num_patches=16,
                 s2_patch_latent_dim=128,
                 s2_refiner=None):
        super().__init__()
        if s1_refiner is None:
            s1_refiner = Patch2DRefiner(
                patch_latent_dim=s1_patch_latent_dim,
                base_channels=128,
                init_size=(4,4),
                final_size=(8,8)  # e.g. 64 points/patch => 16 patches => 1024
            )
        if s2_refiner is None:
            s2_refiner = Patch2DRefiner(
                patch_latent_dim=s2_patch_latent_dim,
                base_channels=256,
                init_size=(8,8),
                final_size=(16,16)  # e.g. 256/patch => 16 patches => 4096
            )

        self.stage1 = MultiPatchStage(
            global_dim=global_dim,
            patch_latent_dim=s1_patch_latent_dim,
            num_patches=s1_num_patches,
            patch_refiner=s1_refiner
        )
        self.stage2 = MultiPatchStage(
            global_dim=global_dim,
            patch_latent_dim=s2_patch_latent_dim,
            num_patches=s2_num_patches,
            patch_refiner=s2_refiner
        )

    def forward(self, global_feat):
        # stage1 => ~1k
        stage1_coords = self.stage1(global_feat)
        # optionally incorporate stage1_coords into a new aggregator
        # for demonstration, just re-use global_feat
        stage2_coords = self.stage2(global_feat)
        return stage2_coords  # => e.g. 4096 final


##############################################################################
# 4) Final Full Model integrating everything
##############################################################################

def freeze_params(m):
    for param in m.parameters():
        param.requires_grad=False

class FullDeepModelWithThreeStages(nn.Module):
    """
    partial_coords -> DeepEncoder -> (opt) transformer -> 3-stage patch decoder with partial skip
    => final coords
    """
    def __init__(self,
                 in_dim=3,
                 encoder_embed_dim=64,
                 global_dim=256,
                 use_transformer=True,
                 transformer_dim=128,
                 transformer_layers=4,
                 nhead=8):
        super().__init__()
        self.use_transformer = use_transformer

        # Deeper encoder
        self.encoder = DeepEncoder(in_dim=in_dim, embed_dim=encoder_embed_dim, global_dim=global_dim)

        # partial embed
        self.partial_embedder = PartialPointEmbedding(in_dim=3, embed_dim=64)

        # optional transformer
        if self.use_transformer:
            self.proj_latent = nn.Linear(global_dim, transformer_dim)
            self.proj_back   = nn.Linear(transformer_dim, global_dim)
            self.transformer = GeometricAwareTransformer(d_model=transformer_dim, nhead=nhead, num_layers=transformer_layers)
        else:
            self.bridge = nn.Identity()

        # 3-stage decoder
        self.decoder = MultiStageDecoderWithSkip(
            global_dim=global_dim,
            partial_dim=64,
            s1_num_patches=16,
            s1_patch_latent_dim=128,
            s2_num_patches=16,
            s2_patch_latent_dim=128,
            s3_num_patches=16,
            s3_patch_latent_dim=128,
        )

    def forward(self, partial_coords):
        B, N, _ = partial_coords.shape

        # encode
        global_feat = self.encoder(partial_coords)  # => [B, global_dim]
        # partial embed
        p_embed = self.partial_embedder(partial_coords)  # => [B,64]

        if self.use_transformer:
            x = self.proj_latent(global_feat).unsqueeze(1)  # => [B,1,transformer_dim]
            coords_fake = torch.zeros(B,1,3, device=partial_coords.device)
            x_out = self.transformer(x, coords_fake)
            x_final = x_out.squeeze(1)
            global_feat = self.proj_back(x_final)

        # decode => final (maybe 16k) coords
        completed = self.decoder(global_feat, p_embed)
        return completed


class FullDeepModel(nn.Module):
    """
    partial_coords -> [DeepEncoder] -> (optional Transformer) -> [Multi-Stage Decoder]
    => final coords
    """
    def __init__(self, 
                 in_dim=3,
                 encoder_embed_dim=64,
                 global_dim=256,
                 use_transformer=False,
                 transformer_dim=128,
                 transformer_layers=4,
                 nhead=8,
                 # multi-stage decoder config
                 s1_num_patches=16,
                 s1_patch_latent_dim=128,
                 s2_num_patches=16,
                 s2_patch_latent_dim=128
                 ):
        super().__init__()
        self.use_transformer = use_transformer

        # 1) deep encoder
        self.encoder = DeepEncoder(in_dim=in_dim, embed_dim=encoder_embed_dim, global_dim=global_dim)

        # 2) optional geometry-aware transformer
        if self.use_transformer:
            self.coords_fc = nn.Sequential(  # to produce d_model coords if needed
                nn.Linear(3, transformer_dim),
                nn.ReLU(True)
            )
            self.proj_latent = nn.Linear(global_dim, transformer_dim)
            self.transformer = GeometricAwareTransformer(d_model=transformer_dim, nhead=nhead, num_layers=transformer_layers)
            self.back_fc = nn.Linear(transformer_dim, global_dim)
        else:
            self.bridge = nn.Identity()  # or an MLP if we want

        # 3) multi-stage decoder
        self.decoder = MultiStageDecoder(
            global_dim=global_dim,
            s1_num_patches=s1_num_patches,
            s1_patch_latent_dim=s1_patch_latent_dim,
            s2_num_patches=s2_num_patches,
            s2_patch_latent_dim=s2_patch_latent_dim
        )

    def forward(self, partial_coords):
        # (a) encode
        latent_enc = self.encoder(partial_coords)  # => [B, global_dim]

        if self.use_transformer:
            # For a single token approach, we can't do geometry-based attention meaningfully.
            # We might produce tokens from partial. For demonstration, let's do a simple approach:
            B, G = latent_enc.shape
            # Suppose we create 1 token => no real multi-head benefit
            x = self.proj_latent(latent_enc).unsqueeze(1) # => [B,1,transformer_dim]
            # Suppose coords is partial or we do random approach
            coords_fake = torch.zeros(B, 1, 3, device=partial_coords.device)
            # run transform
            x_out = self.transformer(x, coords_fake) # => [B,1,transformer_dim]
            x_final = x_out.squeeze(1)
            latent_dec = self.back_fc(x_final)  # => [B,global_dim]
        else:
            latent_dec = self.bridge(latent_enc)  # => [B, global_dim]
        # (b) decode multi-stage => final coords
        completed = self.decoder(latent_dec)  # => e.g. 4096
        return completed


##############################################################################
# 5) Train loop (similar to your snippet, but we swap in FullDeepModel)
##############################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time, os
from tqdm import tqdm

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


def train_s3dis_model():
    # 1) Create dataset
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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Instantiate advanced model
    model = FullDeepModelWithThreeStages(
        in_dim=3,
        encoder_embed_dim=64,
        global_dim=256,
        use_transformer=False,   # set True if you want geometry attention
        transformer_dim=128,
        transformer_layers=4,
        nhead=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 100
    alpha_rep = 0.1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)")
        for i, batch in enumerate(train_iter):
            partial = batch["partial"].to(device)  # [B, N, 3]
            full = batch["full"].to(device)        # [B, N, 3]

            optimizer.zero_grad()
            completed = model(partial)  # e.g. => [B, 4096, 3]

            # Use EMD or Chamfer
            loss_emd = emd_loss_sinkhorn(completed, full, reg=0.1, max_iters=50)
            loss_chamfer, _ = chamfer_distance(completed, full)
            # repulsion optional
            rep_loss_val = repulsion_loss(completed, k=4, threshold=0.02)
            # loss = loss_emd + alpha_rep * rep_loss_val
            loss = loss_emd + loss_chamfer + 0.1 * rep_loss_val

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_iter.set_postfix({"loss": loss.item()})

        epoch_loss_avg = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss_avg:.4f}")

        # Validation
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
        print(f"    Validation Chamfer: {val_loss_avg:.4f}")

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

    print("Training complete!")
    return model


if __name__ == "__main__":
    train_s3dis_model()
