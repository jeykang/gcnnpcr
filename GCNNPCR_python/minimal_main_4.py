# -*- coding: utf-8 -*-

import os
import random
import time
from glob import glob
from typing import List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.utils.data import Dataset
from tqdm import tqdm

# The following imports reference modules that are not present in the trimmed
# version of this repository.  The associated functionality (cross‑attention
# transformer, point cloud augmentation, DGCNN/PointNet2 encoders, combined
# losses and EMD) is reimplemented later in this file.  These imports are
# commented out to avoid ImportError.
# from modules.attention import CrossAttentionTransformer
# from modules.augmentation import augment_pointcloud
# from modules.decoder_refinement import CoarseToFineDecoder
# from modules.dgcnn import DGCNNEncoder
# from modules.losses import CombinedLoss
# from modules.pointnet2 import PointNet2Encoder
# from .emdloss_new import SinkhornEMDLoss

matplotlib.use("Agg")  # Use non‑interactive backend for saving images

# ===========================================================================
# Utilities
# ===========================================================================

def local_knn_coords(coords: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Compute the indices of the k nearest neighbours for every point.

    Args:
        coords: Tensor of shape ``[N, 3]`` containing 3D coordinates.
        k: Number of neighbours.

    Returns:
        Tensor of shape ``[N, k]`` with indices of the k nearest neighbours.
    """
    dist = torch.cdist(coords, coords)
    knn_idx = dist.topk(k, largest=False, dim=1).indices
    return knn_idx


def robust_loadtxt(file_path: str) -> np.ndarray:
    """Read a whitespace‑separated text file containing at least six values
    per line and return a floating point array.  Lines that fail to parse
    are skipped.

    Args:
        file_path: Path to the ``.txt`` file.

    Returns:
        A NumPy array of shape ``[N, 6]`` where ``N`` is the number of
        successfully parsed rows.  If the file is empty or no valid rows
        are found an empty array is returned.
    """
    valid_rows: List[List[float]] = []
    with open(file_path, "r", errors="replace") as f:
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


def compute_normals_pca(coords_t: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Compute approximate normals using PCA of each point's k‑neighbourhood.

    Args:
        coords_t: Tensor of shape ``[N, 3]`` containing 3D coordinates.
        k: Number of neighbours used to estimate the normal.

    Returns:
        Tensor of shape ``[N, 3]`` with unit normals.
    """
    N = coords_t.shape[0]
    knn_idx = local_knn_coords(coords_t, k=k)
    normals = torch.zeros_like(coords_t)
    for i in range(N):
        neighbor_pts = coords_t[knn_idx[i]]
        mean_ = neighbor_pts.mean(dim=0, keepdim=True)
        cov_ = (neighbor_pts - mean_).t() @ (neighbor_pts - mean_)
        eigvals, eigvecs = torch.linalg.eigh(cov_)
        normal_i = eigvecs[:, 0]
        normals[i] = normal_i
    normals = F.normalize(normals, dim=-1)
    return normals

# ---------------------------------------------------------------------------
# Data augmentation and loss functions
# ---------------------------------------------------------------------------

def augment_pointcloud(pc: torch.Tensor,
                       sigma: float = 0.01,
                       clip: float = 0.05,
                       scale_low: float = 0.9,
                       scale_high: float = 1.1,
                       drop_ratio: float = 0.1) -> torch.Tensor:
    """
    Apply simple augmentations to a batch of point clouds.  The first three
    channels of ``pc`` are assumed to be XYZ coordinates and the remaining
    channels (if any) are treated as features.  Augmentations include jitter,
    random scaling, rotation about the Z axis and random point drop‑out.  All
    augmentations are applied identically across each point cloud in the batch.

    Args:
        pc: Tensor of shape ``[B, N, C]`` containing point clouds.  At least
            three channels are required.
        sigma: Standard deviation of Gaussian jitter.
        clip: Maximum absolute value of jitter.
        scale_low: Lower bound of random scaling factor.
        scale_high: Upper bound of random scaling factor.
        drop_ratio: Fraction of points to randomly drop (set to zero).

    Returns:
        Augmented point cloud of the same shape as the input.
    """
    coords = pc[..., :3]
    feats = pc[..., 3:] if pc.shape[-1] > 3 else None
    B, N, _ = coords.shape
    # Jitter
    noise = torch.randn_like(coords) * sigma
    noise = noise.clamp(-clip, clip)
    coords = coords + noise
    # Scaling
    scales = torch.empty(B, 1, 1, device=coords.device).uniform_(scale_low, scale_high)
    coords = coords * scales
    # Rotation around Z axis
    # Generate random angles for each batch
    angles = torch.rand(B, device=coords.device) * 2 * np.pi
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    rot_mats = torch.zeros(B, 3, 3, device=coords.device)
    rot_mats[:, 0, 0] = cos_vals
    rot_mats[:, 0, 1] = -sin_vals
    rot_mats[:, 1, 0] = sin_vals
    rot_mats[:, 1, 1] = cos_vals
    rot_mats[:, 2, 2] = 1.0
    coords = torch.bmm(coords, rot_mats)
    # Random drop‑out
    if drop_ratio > 0.0:
        drop_n = max(1, int(N * drop_ratio))
        for b in range(B):
            idx = torch.randperm(N, device=coords.device)[:drop_n]
            coords[b, idx] = 0.0
            if feats is not None:
                feats[b, idx] = 0.0
    # Concatenate coordinates and features back together
    if feats is not None:
        return torch.cat([coords, feats], dim=-1)
    return coords

def chamfer_distance(pcd1: torch.Tensor, pcd2: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric Chamfer distance between two batches of point clouds.

    Args:
        pcd1: Tensor of shape ``[B, N, 3]``.
        pcd2: Tensor of shape ``[B, M, 3]``.

    Returns:
        Scalar tensor containing the average Chamfer distance across the batch.
    """
    dist = torch.cdist(pcd1, pcd2)  # [B,N,M]
    min1 = dist.min(dim=2)[0]  # [B,N]
    min2 = dist.min(dim=1)[0]  # [B,M]
    cd = min1.mean(dim=1) + min2.mean(dim=1)
    return cd.mean()

def repulsion_loss(pcd: torch.Tensor, k: int = 4, threshold: float = 0.02) -> torch.Tensor:
    """
    Compute a simple repulsion loss that penalises pairs of points within a
    threshold distance.  For each point the ``k`` nearest neighbours are
    considered; distances smaller than ``threshold`` contribute to the loss.

    Args:
        pcd: Tensor of shape ``[B, N, 3]`` containing point coordinates.
        k: Number of nearest neighbours to inspect for each point.
        threshold: Repulsion distance threshold.

    Returns:
        Scalar tensor containing the average repulsion penalty.
    """
    B, N, _ = pcd.shape
    loss = 0.0
    for b in range(B):
        dist = torch.cdist(pcd[b], pcd[b])  # [N,N]
        # Add large constant to diagonal to avoid self‑interaction
        dist = dist + torch.eye(N, device=pcd.device) * 1e9
        knn_vals, _ = dist.topk(k, largest=False)
        mask = knn_vals < threshold
        # Penalise distances below the threshold
        penal = (threshold - knn_vals[mask]) ** 2
        loss = loss + penal.mean()
    return loss / B

def combined_loss(pred: torch.Tensor, gt: torch.Tensor,
                  cd_w: float = 1.0, rep_w: float = 0.1) -> torch.Tensor:
    """
    Combined loss composed of Chamfer distance and repulsion penalty.  Normal
    consistency can optionally be added by extending this function.

    Args:
        pred: Predicted point clouds of shape ``[B, N, 3]``.
        gt: Ground‑truth point clouds of shape ``[B, M, 3]``.
        cd_w: Weight for the Chamfer distance term.
        rep_w: Weight for the repulsion term.

    Returns:
        Scalar loss tensor.
    """
    cd = chamfer_distance(pred, gt)
    rep = repulsion_loss(pred)
    return cd_w * cd + rep_w * rep


class S3DISDataset(Dataset):
    """Patch‑based dataset for the Stanford 3D Indoor Scenes (S3DIS) dataset.

    Instead of sampling an entire room down to ``num_points`` points, this
    loader extracts multiple local patches from each room.  For each room
    ``patches_per_room`` patches are generated by choosing a random centre
    within the room's bounding box and selecting the ``num_points`` points
    nearest to that centre.  This encourages the network to learn both local
    detail and global context, as recommended in the accompanying report.
    """

    def __init__(self,
                 root: str,
                 mask_ratio: float = 0.5,
                 num_points: int = 8192,
                 split: str = "train",
                 normal_k: int = 16,
                 patches_per_room: int = 1,
                 train_areas: Optional[List[str]] = None,
                 test_areas: Optional[List[str]] = None) -> None:
        super().__init__()
        self.root = root
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.split = split
        self.normal_k = normal_k
        self.patches_per_room = max(1, patches_per_room)

        # Gather all room‑level ``.txt`` files, excluding annotation files and
        # alignment metadata.  ``Annotations`` folders contain object‑level
        # point clouds, which are not used for training.
        pattern = os.path.join(root, "**", "*.txt")
        all_files: List[str] = [f for f in glob(pattern, recursive=True)
                                if ("alignmentAngle" not in f and "Annotations" not in f)]

        # Use explicit area‑level splits if specified.  The S3DIS dataset
        # consists of six areas (Area_1 through Area_6).  Many papers train
        # on five areas and evaluate on the remaining one; e.g. training on
        # Areas 1–5 and testing on Area 6.  You can override this behaviour by
        # passing `train_areas` and `test_areas` lists; otherwise an 80/20
        # random split is applied.
        if train_areas or test_areas:
            # Normalise area names to lower‑case to avoid mismatches
            train_set = {a.lower() for a in (train_areas or [])}
            test_set = {a.lower() for a in (test_areas or [])}
            train_files: List[str] = []
            test_files: List[str] = []
            for f in all_files:
                # Extract area name by splitting path; expects "Area_X" to
                # appear in the file path.
                parts = f.replace("\\", "/").split("/")
                area = next((p for p in parts if p.lower().startswith("area_")), "")
                area_l = area.lower()
                if area_l in train_set:
                    train_files.append(f)
                elif area_l in test_set:
                    test_files.append(f)
            # If either list is empty, fallback to a simple 80/20 split.  This
            # ensures that the dataset always has at least one sample.  The
            # fallback will be triggered when the specified area names do not
            # appear in the directory structure.
            if not train_files or not test_files:
                split_idx = int(0.8 * len(all_files))
                if self.split == "train":
                    self.files = all_files[:split_idx]
                else:
                    self.files = all_files[split_idx:]
            else:
                self.files = train_files if self.split == "train" else test_files
        else:
            # 80/20 split by file order when area lists are not provided
            split_idx = int(0.8 * len(all_files))
            if self.split == "train":
                self.files = all_files[:split_idx]
            else:
                self.files = all_files[split_idx:]

    def __len__(self) -> int:
        # Each file yields ``patches_per_room`` samples.
        return len(self.files) * self.patches_per_room

    def __getitem__(self, idx: int) -> dict:
        # Determine which file and which patch are requested.
        file_idx = idx // self.patches_per_room
        file_path = self.files[file_idx]
        arr = robust_loadtxt(file_path)
        if arr.size == 0:
            arr = np.zeros((self.num_points, 6), dtype=np.float32)
        coords_full = arr[:, :3]
        N = coords_full.shape[0]

        # If the room has fewer points than ``num_points``, sample with replacement.
        if N <= self.num_points:
            indices = np.random.choice(N, self.num_points, replace=True)
        else:
            # Choose a random centre within the room's axis‑aligned bounding box
            min_c = coords_full.min(axis=0)
            max_c = coords_full.max(axis=0)
            patch_center = min_c + np.random.rand(3) * (max_c - min_c)
            # Compute squared distances to the centre and select nearest ``num_points`` points
            dist2 = np.sum((coords_full - patch_center) ** 2, axis=1)
            nearest_indices = np.argpartition(dist2, self.num_points)[:self.num_points]
            indices = nearest_indices
        sample = arr[indices]  # [num_points, 6]
        coords = sample[:, :3]

        # Normalise the patch to the unit cube centred at the origin
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        center = (min_c + max_c) / 2.0
        scale = (max_c - min_c).max() / 2.0
        if scale < 1e-8:
            scale = 1.0
        coords = (coords - center) / scale

        coords_t = torch.from_numpy(coords).float()
        # Estimate normals using PCA
        normals_t = compute_normals_pca(coords_t, k=self.normal_k)

        full_6d = torch.cat([coords_t, normals_t], dim=-1)  # [num_points,6]
        # Randomly mask a cube of points to simulate occlusions
        mask_count = int(self.num_points * self.mask_ratio)
        mask_idx = np.random.choice(self.num_points, mask_count, replace=False)
        partial_6d = full_6d.clone()
        partial_6d[mask_idx, :] = 0.0

        return {
            "partial": partial_6d,
            "full": full_6d
        }


# ===========================================================================
# Graph Neural Network Encoder
# ===========================================================================
from torch_geometric.nn import GCNConv, knn_graph


class GraphEncoder(nn.Module):
    """
    Graph convolutional encoder for point clouds.  This class supports two
    variants of edge aggregation: a standard graph convolution (GCN) and a
    graph attention (GAT) layer.  Attention can better capture the relative
    importance of neighbours within a local neighbourhood, which is inspired
    by the relation‑based weighting mechanism proposed in PointCFormer【538684515318844†L43-L57】.

    Args:
        in_dim: Input feature dimensionality (e.g. 6 for XYZ + normals).
        hidden_dims: List of hidden feature dimensions for successive layers.
        out_dim: Output feature dimensionality per point.
        k: Number of nearest neighbours used to construct the graph.
        use_attention: If True, attempt to use a GAT layer instead of a GCN.
        heads: Number of attention heads when ``use_attention`` is True.
    """

    def __init__(self,
                 in_dim: int = 6,
                 hidden_dims: Optional[List[int]] = None,
                 out_dim: int = 128,
                 k: int = 16,
                 use_attention: bool = False,
                 heads: int = 4) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        self.k = k
        self.use_attention = use_attention
        self.heads = heads
        # Attempt to import GATConv if attention is requested
        if self.use_attention:
            try:
                from torch_geometric.nn import GATConv  # type: ignore
                # When using attention, ``concat=False`` ensures that the output
                # dimension is ``out_c`` rather than ``out_c * heads``.  Do not
                # divide by the number of heads here; the convolution will
                # internally split and average across heads.  Passing
                # ``out_c//self.heads`` (as in a previous version) reduced the
                # feature dimension and caused a mismatch with the subsequent
                # layers.  See https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html
                # for details.
                self.conv_cls = lambda in_c, out_c: GATConv(in_c, out_c, heads=self.heads, concat=False)
            except Exception:
                # Fallback to GCNConv if GAT is unavailable
                self.use_attention = False
        if not self.use_attention:
            from torch_geometric.nn import GCNConv  # type: ignore
            self.conv_cls = lambda in_c, out_c: GCNConv(in_c, out_c)
        self.gconvs = nn.ModuleList()
        prev_dim = in_dim
        for hd in hidden_dims:
            self.gconvs.append(self.conv_cls(prev_dim, hd))
            prev_dim = hd
        self.final_lin = nn.Linear(prev_dim, out_dim)

    def forward(self, feats_batch: torch.Tensor) -> torch.Tensor:
        """
        Process a batch of point clouds.  ``feats_batch`` should have shape
        ``[B, N, C]`` where the first three channels correspond to coordinates
        used to build the k‑NN graph.  Any additional channels are treated as
        per‑point features and are passed through the convolution layers.

        Returns:
            A tensor of shape ``[B, N, out_dim]`` containing per‑point features.
        """
        B, N, C = feats_batch.shape
        outputs: List[torch.Tensor] = []
        for b in range(B):
            feats_b = feats_batch[b]  # [N,C]
            coords_3d = feats_b[:, :3]  # [N,3]
            edge_idx = knn_graph(coords_3d, k=self.k, loop=False)
            x = feats_b
            for conv in self.gconvs:
                x = conv(x, edge_idx)
                x = F.relu(x)
            x_out = self.final_lin(x)
            outputs.append(x_out.unsqueeze(0))
        return torch.cat(outputs, dim=0)


# ===========================================================================
# Geometry‑Aware Transformer
# ===========================================================================

class GeomAttention(nn.Module):
    """Self‑attention layer with a simple geometric bias based on
    pairwise dot products of coordinates.  This design is inspired by
    SnowflakeNet but simplified for clarity.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8) -> None:
        super().__init__()
        self.nhead = nhead
        self.dk = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply geometry‑aware self‑attention.

        Args:
            x: Tensor of shape ``[B,N,d_model]`` containing token features.
            coords: Tensor of shape ``[B,N,3]`` containing 3‑D coordinates.

        Returns:
            Tensor of shape ``[B,N,d_model]`` with updated features.
        """
        B, N, D = x.shape
        Q = self.w_q(x).view(B, N, self.nhead, self.dk).permute(0, 2, 1, 3)
        K = self.w_k(x).view(B, N, self.nhead, self.dk).permute(0, 2, 1, 3)
        V = self.w_v(x).view(B, N, self.nhead, self.dk).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)  # [B,nhead,N,N]

        # Geometry bias based on dot products
        coords_2d = coords.view(B * N, 3)
        dot_mat = torch.matmul(coords_2d, coords_2d.T)  # [B*N, B*N]
        G = torch.zeros((B, N, N), device=x.device)
        for b in range(B):
            st = b * N
            ed = (b + 1) * N
            G[b] = dot_mat[st:ed, st:ed]
        G = self.alpha * G
        G = G.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        scores = scores + G

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # [B,nhead,N,dk]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.nhead * self.dk)
        return out


class GeomMultiTokenTransformer(nn.Module):
    """Stack of geometry‑aware attention layers with feed‑forward networks.
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GeomAttention(d_model, nhead)
            ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(True),
                nn.Linear(4 * d_model, d_model)
            )
            self.layers.append(nn.ModuleList([attn, ff]))
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        for i, (attn, ffn) in enumerate(self.layers):
            x_attn = attn(x, coords)
            x = x + x_attn
            x = self.norm1[i](x)
            x_ff = ffn(x)
            x = x + x_ff
            x = self.norm2[i](x)
        return x


# ===========================================================================
# Snowflake‑like Decoder
# ===========================================================================


def local_knn(partial: torch.Tensor, predicted: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Retrieve k nearest neighbours from the partial input for each predicted point.
    """
    device = partial.device
    B, M, _ = partial.shape
    _, N, _ = predicted.shape
    out_list: List[torch.Tensor] = []
    for b in range(B):
        part_b = partial[b, :, :3]   # [M,3]
        pred_b = predicted[b, :, :3] # [N,3]
        dist = torch.cdist(pred_b, part_b, p=2)
        knn_idx = dist.topk(k, largest=False, dim=1).indices
        neigh_b: List[torch.Tensor] = []
        for i in range(N):
            row_idx = knn_idx[i]
            neigh_b.append(part_b[row_idx])
        neigh_b = torch.stack(neigh_b, dim=0)  # [N,k,3]
        out_list.append(neigh_b.unsqueeze(0))
    return torch.cat(out_list, dim=0)  # [B,N,k,3]


class MLP_Res(nn.Module):
    """Residual 1D convolution block used in the decoder.
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x)
        out = self.conv2(F.relu(self.conv1(x))) + sc
        return out


def fps_subsample(pcd_xyz: torch.Tensor, out_n: int) -> torch.Tensor:
    """Naive farthest point sampling (random fallback)."""
    B, N, _ = pcd_xyz.shape
    device = pcd_xyz.device
    if N <= out_n:
        return pcd_xyz
    idx = torch.stack([torch.randperm(N, device=device)[:out_n] for _ in range(B)], dim=0)
    idx_expand = idx.unsqueeze(-1).expand(-1, -1, 3)
    out = torch.gather(pcd_xyz, 1, idx_expand)
    return out


class SPDStage(nn.Module):
    """One refinement stage for the decoder."""

    def __init__(self, in_feat_dim: int = 128, radius: float = 1.0, bounding: bool = True, up_factor: int = 2, stage_idx: int = 0) -> None:
        super().__init__()
        self.radius = radius
        self.bounding = bounding
        self.up_factor = up_factor
        self.stage_idx = stage_idx
        self.in_feat_dim = in_feat_dim
        self.mlp_pcd = nn.Conv1d(3, 64, 1)
        self.mlp_merge = nn.Conv1d(64 + in_feat_dim, in_feat_dim, 1)
        self.deconv_feat = nn.ConvTranspose1d(in_feat_dim, in_feat_dim, kernel_size=up_factor, stride=up_factor)
        self.mlp_offset = MLP_Res(in_dim=in_feat_dim, hidden_dim=in_feat_dim, out_dim=in_feat_dim)
        self.conv_offset = nn.Conv1d(in_feat_dim, 3, 1)

    def forward(self, pcd_xyz: torch.Tensor, K_prev: torch.Tensor, global_feat: torch.Tensor, partial_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, N = pcd_xyz.shape
        local_feat = self.mlp_pcd(pcd_xyz)
        if K_prev is None:
            B0, c_l, N0 = local_feat.shape
            K_prev = local_feat.new_zeros((B0, self.in_feat_dim, N0))
        feats_merge = torch.cat([local_feat, K_prev], dim=1)
        feats_merge = self.mlp_merge(feats_merge)
        gf_broad = global_feat.repeat(1, 1, N)
        feats_merge = feats_merge + gf_broad
        K_up = self.deconv_feat(feats_merge)
        offset_feat = self.mlp_offset(K_up)
        delta = self.conv_offset(offset_feat)
        if self.bounding:
            delta = torch.tanh(delta) / (self.radius ** (self.stage_idx))
        pcd_up = pcd_xyz.repeat_interleave(self.up_factor, dim=2)
        pcd_up = pcd_up + delta
        K_curr = offset_feat
        return pcd_up, K_curr


class SPDBasedDecoder(nn.Module):
    """Multi‑stage decoder that upsamples a coarse seed to a dense point cloud."""

    def __init__(self, in_feat_dim: int = 128, coarse_num: int = 64, up_factors: List[int] = [2, 2, 2, 2, 2, 2, 2], bounding: bool = True, radius: float = 1.0) -> None:
        super().__init__()
        self.coarse_num = coarse_num
        self.init_fc = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim),
            nn.ReLU(True),
            nn.Linear(in_feat_dim, 3 * coarse_num)
        )
        self.stages = nn.ModuleList()
        stage_idx = 0
        for uf in up_factors:
            spd_stage = SPDStage(in_feat_dim=in_feat_dim, radius=radius, bounding=bounding, up_factor=uf, stage_idx=stage_idx)
            self.stages.append(spd_stage)
            stage_idx += 1

    def forward(self, partial_6d: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        B, N, _ = partial_6d.shape
        fc_out = self.init_fc(global_feat.squeeze(-1))
        pcd_coarse = fc_out.view(B, 3, self.coarse_num)
        K = None
        pcd = pcd_coarse
        for spd in self.stages:
            pcd, K = spd(pcd, K, global_feat, partial_6d[..., :3])
        return pcd


class FullModelSnowflake(nn.Module):
    """
    Complete model combining an encoder, a geometry‑aware transformer and a
    Snowflake‑style decoder.  The encoder is a configurable graph network
    (either GCN or GAT) defined in this file; the transformer is the
    ``GeomMultiTokenTransformer``; and the decoder is ``SPDBasedDecoder``.

    Args:
        g_hidden_dims: Hidden dimensions for the graph encoder.
        g_out_dim: Output dimension of the graph encoder.
        t_d_model: Token dimension for the transformer and decoder.
        t_nhead: Number of heads in the transformer.
        t_layers: Number of transformer layers.
        coarse_num: Number of coarse seed points output by the decoder.
        use_attention_encoder: Whether to use graph attention in the encoder.
        bounding: Whether to bound point offsets in the decoder.
        radius: Radius parameter controlling decoder offset scaling.
    """

    def __init__(self,
                 g_hidden_dims: Optional[List[int]] = None,
                 g_out_dim: int = 128,
                 t_d_model: int = 128,
                 t_nhead: int = 8,
                 t_layers: int = 4,
                 coarse_num: int = 64,
                 use_attention_encoder: bool = False,
                 bounding: bool = True,
                 radius: float = 1.0) -> None:
        super().__init__()
        if g_hidden_dims is None:
            g_hidden_dims = [64, 128]
        # Encoder: GraphEncoder with optional attention
        self.encoder = GraphEncoder(in_dim=6,
                                    hidden_dims=g_hidden_dims,
                                    out_dim=g_out_dim,
                                    k=16,
                                    use_attention=use_attention_encoder,
                                    heads=4)
        # Transformer: geometry‑aware multi‑token transformer
        self.transformer = GeomMultiTokenTransformer(d_model=t_d_model,
                                                     nhead=t_nhead,
                                                     num_layers=t_layers)
        # Bridge layer maps encoder output dimension to transformer dimension
        self.bridge = nn.Linear(g_out_dim, t_d_model) if g_out_dim != t_d_model else nn.Identity()
        # Decoder: Snowflake‑style upsampling network
        self.decoder = SPDBasedDecoder(in_feat_dim=t_d_model,
                                       coarse_num=coarse_num,
                                       up_factors=[2] * 7,
                                       bounding=bounding,
                                       radius=radius)

    def forward(self, partial_6d: torch.Tensor) -> torch.Tensor:
        # Apply augmentation during training
        if self.training:
            partial_6d = augment_pointcloud(partial_6d)
        # Encode per‑point features
        tokens = self.encoder(partial_6d)  # [B,N,g_out_dim]
        # Map to transformer dimension
        x = self.bridge(tokens)  # [B,N,t_d_model]
        # Apply transformer with geometry bias
        x = self.transformer(x, partial_6d[..., :3])  # coords used for bias
        # Aggregate global feature by averaging token features
        global_feat = x.mean(dim=1, keepdim=True).permute(0, 2, 1)  # [B,t_d_model,1]
        # Decode to full point cloud
        return self.decoder(partial_6d, global_feat)  # returns [B,3,num_points]


# ===========================================================================
# Visualisation and Training
# ===========================================================================


def save_point_cloud_comparison(partial: torch.Tensor, completed: torch.Tensor, original: torch.Tensor, epoch: int, out_dir: str = "visuals") -> None:
    """Save a comparison of partial, completed, and original point clouds as an image."""
    os.makedirs(out_dir, exist_ok=True)
    partial_np = partial.detach().cpu().numpy()
    completed_np = completed.detach().cpu().numpy()
    original_np = original.detach().cpu().numpy()
    all_points = np.concatenate([partial_np, completed_np, original_np], axis=0)
    min_xyz = all_points.min(axis=0)
    max_xyz = all_points.max(axis=0)
    eps = 1e-5
    range_xyz = max_xyz - min_xyz
    range_xyz[range_xyz < eps] = eps
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(partial_np[:, 0], partial_np[:, 1], partial_np[:, 2], s=1, c='r')
    ax1.set_title("Partial (Masked)")
    ax1.set_xlim3d(min_xyz[0], max_xyz[0])
    ax1.set_ylim3d(min_xyz[1], max_xyz[1])
    ax1.set_zlim3d(min_xyz[2], max_xyz[2])
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(completed_np[:, 0], completed_np[:, 1], completed_np[:, 2], s=1, c='b')
    ax2.set_title("Completed (Predicted)")
    ax2.set_xlim3d(min_xyz[0], max_xyz[0])
    ax2.set_ylim3d(min_xyz[1], max_xyz[1])
    ax2.set_zlim3d(min_xyz[2], max_xyz[2])
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(original_np[:, 0], original_np[:, 1], original_np[:, 2], s=1, c='g')
    ax3.set_title("Original (Unmasked)")
    ax3.set_xlim3d(min_xyz[0], max_xyz[0])
    ax3.set_ylim3d(min_xyz[1], max_xyz[1])
    ax3.set_zlim3d(min_xyz[2], max_xyz[2])
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"completion_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved point cloud comparison to {save_path}")


def train_s3dis_model() -> FullModelSnowflake:
    """Train the model on the S3DIS dataset using the patch‑based loader.

    This function attempts to locate the S3DIS dataset by checking a default
    path on the ``E:`` drive and, if that does not exist, a fallback on the
    ``F:`` drive.  Users may customise the path by setting the ``S3DIS_ROOT``
    environment variable before running this script.  If neither path is
    found, a ``FileNotFoundError`` is raised to signal that the dataset
    location must be specified.
    """
    # Determine the dataset root path.  Prefer an environment override
    # (useful for running on different machines) but fallback to typical
    # Windows drive letters used in previous experiments.  Using raw string
    # literals avoids escape issues on Windows paths.
    env_root = os.environ.get("S3DIS_ROOT")
    if env_root and os.path.exists(env_root):
        dataset_root = env_root
    else:
        # Check typical locations on E: and F: drives.  The default path used
        # previously was ``E:\\S3DIS\\cvg-data.inf.ethz.ch\\s3dis``.  Some
        # installations (as indicated by the supplied directory listing)
        # reside on the F: drive instead.  Test both and pick the first one
        # that exists.
        default_paths = [
            r"E:\\S3DIS\\cvg-data.inf.ethz.ch\\s3dis",
            r"F:\\S3DIS\\cvg-data.inf.ethz.ch\\s3dis",
        ]
        dataset_root = None
        for candidate in default_paths:
            if os.path.exists(candidate):
                dataset_root = candidate
                break
    if not dataset_root:
        raise FileNotFoundError(
            "Could not locate the S3DIS dataset. Please set the S3DIS_ROOT "
            "environment variable or install the dataset at E:/S3DIS or F:/S3DIS."
        )

    # Define area splits.  By default we train on Areas 1–5 and validate on Area 6.
    train_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"]
    test_areas = ["Area_6"]

    # Instantiate datasets with multiple patches per room to improve context coverage.
    train_dataset = S3DISDataset(
        root=dataset_root,
        mask_ratio=0.5,
        num_points=8192,
        split="train",
        normal_k=16,
        patches_per_room=4,
        train_areas=train_areas,
        test_areas=test_areas,
    )
    val_dataset = S3DISDataset(
        root=dataset_root,
        mask_ratio=0.5,
        num_points=8192,
        split="val",
        normal_k=16,
        patches_per_room=4,
        train_areas=train_areas,
        test_areas=test_areas,
    )

    # Build data loaders.  Use a batch size of 1 due to memory constraints.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    # Configure device and training hyper‑parameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    model = FullModelSnowflake(
        g_hidden_dims=[64, 128],
        g_out_dim=128,
        t_d_model=128,
        t_nhead=8,
        t_layers=4,
        coarse_num=64,
        use_attention_encoder=True,
        radius=1.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for _, batch in enumerate(train_iter):
            partial_6d = batch["partial"].to(device)
            full_6d = batch["full"].to(device)
            optimizer.zero_grad()
            completed = model(partial_6d)  # [B,3,8192]
            completed_coords = completed.permute(0, 2, 1).contiguous()  # [B,8192,3]
            gt_coords = full_6d[..., :3]
            loss = combined_loss(completed_coords, gt_coords)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_iter.set_postfix({"loss": loss.item()})
        scheduler.step()
        epoch_loss_avg = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_loss_avg:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_val in val_loader:
                partial_val_6d = batch_val["partial"].to(device)
                full_val_6d = batch_val["full"].to(device)
                completed_val = model(partial_val_6d)
                completed_val_coords = completed_val.permute(0, 2, 1).contiguous()
                gt_val_coords = full_val_6d[..., :3]
                val_loss = val_loss + chamfer_distance(completed_val_coords, gt_val_coords).item()
        val_loss_avg = val_loss / len(val_loader)
        print(f"    Validation Chamfer: {val_loss_avg:.4f}")
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved model checkpoint to {ckpt_path}")
        # Visualise sample output using the first training sample of this epoch
        partial_0 = partial_6d[0, ..., :3].detach().cpu()
        completed_0 = completed_coords[0].detach().cpu()
        original_0 = full_6d[0, ..., :3].detach().cpu()
        save_point_cloud_comparison(partial_0, completed_0, original_0, epoch+1)
    return model


if __name__ == "__main__":
    train_s3dis_model()