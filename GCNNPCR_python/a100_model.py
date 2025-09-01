#!/usr/bin/env python
"""
Multi-GPU optimized model for point cloud completion
Designed to leverage 8x NVIDIA A100 setup effectively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import math

# Import base components
from minimal_main_4 import GraphEncoder, local_knn
from enhanced_model import AttentionPropagation


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for better spatial awareness"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        B, N, C = x.shape
        return x + self.pe[:N].unsqueeze(0).expand(B, -1, -1)


class MultiScaleEncoder(nn.Module):
    """Multi-scale feature extraction optimized for large batch processing"""
    def __init__(self, in_dim: int = 6, hidden_dims: List[int] = [128, 256, 512], 
                 out_dim: int = 512, scales: List[int] = [512, 1024, 2048]):
        super().__init__()
        self.scales = scales
        
        # Multiple graph encoders at different scales
        self.encoders = nn.ModuleList([
            GraphEncoder(in_dim, hidden_dims[:2], hidden_dims[1], k=16),
            GraphEncoder(in_dim, hidden_dims[1:], hidden_dims[2], k=32),
            GraphEncoder(in_dim, hidden_dims, out_dim, k=64)
        ])
        
        # Feature fusion with GroupNorm (more stable than BatchNorm for variable batch sizes)
        fusion_dim = sum([hidden_dims[1], hidden_dims[2], out_dim])
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, out_dim),
            nn.GroupNorm(32, out_dim),  # GroupNorm with 32 groups
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.GroupNorm(32, out_dim),
            nn.ReLU(inplace=True)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        features = []
        
        # Extract features at multiple scales
        for scale, encoder in zip(self.scales, self.encoders):
            if N > scale:
                # Farthest point sampling for downsampling
                idx = self.fps_sampling(x[..., :3], scale)
                x_scaled = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
            else:
                x_scaled = x
            
            feat = encoder(x_scaled)  # [B, scale, hidden_dim]
            
            # Upsample back to original size if needed
            if feat.shape[1] < N:
                feat = self.upsample_features(feat, x[..., :3], x_scaled[..., :3])
            
            features.append(feat)
        
        # Fuse multi-scale features
        fused = torch.cat(features, dim=-1)  # [B, N, fusion_dim]
        output = self.fusion(fused)  # [B, N, out_dim]
        
        # Add positional encoding
        output = self.pos_encoding(output)
        
        return output
    
    def fps_sampling(self, xyz: torch.Tensor, num_points: int) -> torch.Tensor:
        """Farthest point sampling"""
        B, N, _ = xyz.shape
        device = xyz.device
        
        centroids = torch.zeros(B, num_points, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(num_points):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=1)[1]
        
        return centroids
    
    def upsample_features(self, feat_sparse: torch.Tensor, xyz_dense: torch.Tensor, 
                          xyz_sparse: torch.Tensor) -> torch.Tensor:
        """Upsample features using k-NN interpolation"""
        B, N_dense, _ = xyz_dense.shape
        B, N_sparse, C = feat_sparse.shape
        
        # Find 3 nearest neighbors for interpolation
        dist = torch.cdist(xyz_dense, xyz_sparse)  # [B, N_dense, N_sparse]
        knn_dist, knn_idx = dist.topk(3, largest=False, dim=-1)  # [B, N_dense, 3]
        
        # Inverse distance weighting
        weights = 1.0 / (knn_dist + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Gather features and interpolate
        feat_upsampled = []
        for b in range(B):
            feat_knn = feat_sparse[b][knn_idx[b]]  # [N_dense, 3, C]
            feat_weighted = (feat_knn * weights[b].unsqueeze(-1)).sum(dim=1)  # [N_dense, C]
            feat_upsampled.append(feat_weighted)
        
        return torch.stack(feat_upsampled, dim=0)


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that prevents collapse through progressive refinement"""
    def __init__(self, feat_dim: int = 512, num_stages: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_stages = num_stages
        
        # Initial points are learned parameters (better than random)
        self.init_points = nn.Parameter(torch.randn(256, 3) * 0.1)
        
        # Progressive upsampling stages: 256 -> 512 -> 1024 -> 2048 -> 4096 -> 8192
        self.stages = nn.ModuleList()
        points = 256
        for i in range(num_stages):
            self.stages.append(AdaptiveRefinementStage(
                feat_dim=feat_dim,
                num_points=points,
                next_points=points * 2,
                stage_idx=i
            ))
            points *= 2
        
        # Attention-based feature propagation between stages
        self.propagations = nn.ModuleList([
            AttentionPropagation(feat_dim) for _ in range(num_stages)
        ])
        
        # Coordinate prediction heads with residual connections
        self.coord_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GroupNorm(32, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, 3)
            ) for _ in range(num_stages)
        ])
    
    def forward(self, partial_xyz: torch.Tensor, partial_feat: torch.Tensor,
                global_feat: torch.Tensor) -> torch.Tensor:
        B = global_feat.shape[0]
        device = global_feat.device
        
        # Initialize with learned seed points
        xyz = self.init_points.unsqueeze(0).expand(B, -1, -1).to(device)
        feat = global_feat.unsqueeze(1).expand(B, 256, -1)
        
        # Progressive refinement
        for i, (stage, prop, coord_head) in enumerate(
            zip(self.stages, self.propagations, self.coord_heads)):
            
            # Refine and upsample
            xyz_new, feat_new = stage(xyz, feat, global_feat)
            
            # Propagate features from partial input
            feat_new = prop(partial_xyz, partial_feat, xyz_new, feat_new)
            
            # Predict coordinate offsets with residual
            offset = coord_head(feat_new) * (0.1 / (i + 1))  # Decreasing offset scale
            xyz_new = xyz_new + offset
            
            # Apply repulsion to prevent clustering
            if i < self.num_stages - 1:  # Not on final stage
                xyz_new = self.apply_repulsion(xyz_new, min_dist=0.01 / (2 ** i))
            
            xyz = xyz_new
            feat = feat_new
        
        return xyz
    
    def apply_repulsion(self, xyz: torch.Tensor, min_dist: float = 0.01) -> torch.Tensor:
        """Apply repulsion force to prevent point clustering"""
        B, N, _ = xyz.shape
        
        # Compute pairwise distances
        dist = torch.cdist(xyz, xyz) + torch.eye(N, device=xyz.device) * 1e6
        
        # Find points that are too close
        too_close = (dist < min_dist).float()
        
        # Compute repulsion forces
        if too_close.any():
            diff = xyz.unsqueeze(2) - xyz.unsqueeze(1)  # [B, N, N, 3]
            forces = diff / (dist.unsqueeze(-1) + 1e-8)  # Normalize by distance
            forces = forces * too_close.unsqueeze(-1)  # Apply only where too close
            total_force = forces.sum(dim=2)  # Sum forces from all neighbors
            
            # Apply forces with small step
            xyz = xyz + total_force * 0.01
        
        return xyz


class AdaptiveRefinementStage(nn.Module):
    """Single refinement stage with adaptive upsampling"""
    def __init__(self, feat_dim: int, num_points: int, next_points: int, stage_idx: int):
        super().__init__()
        self.num_points = num_points
        self.next_points = next_points
        self.stage_idx = stage_idx
        
        # Feature refinement with GroupNorm
        self.refine = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive splitting network
        self.split = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GroupNorm(32, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 6)  # 3 for offset, 3 for second point offset
        )
    
    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, 
                global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        
        # Combine with global features
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([feat, global_expanded], dim=-1)
        
        # Refine features
        feat_refined = self.refine(combined)
        
        # Generate new points through adaptive splitting
        split_params = self.split(combined)
        offset1 = split_params[..., :3] * (0.05 / (self.stage_idx + 1))
        offset2 = split_params[..., 3:] * (0.05 / (self.stage_idx + 1))
        
        # Create two points from each original point
        xyz1 = xyz + offset1
        xyz2 = xyz + offset2
        xyz_new = torch.cat([xyz1, xyz2], dim=1)  # [B, N*2, 3]
        
        # Duplicate features
        feat_new = feat_refined.repeat(1, 2, 1)  # [B, N*2, feat_dim]
        
        return xyz_new, feat_new


class DiscriminatorNetwork(nn.Module):
    """Discriminator for adversarial training (optional, uses extra GPU memory)"""
    def __init__(self, num_points: int = 8192, feat_dim: int = 512):
        super().__init__()
        
        # PointNet-style feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, feat_dim, 1)
        )
        
        # Global feature aggregation
        self.global_feat = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
    
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3] point cloud
        Returns:
            score: [B, 1] realness score
        """
        # Transpose for Conv1d
        x = xyz.transpose(1, 2)  # [B, 3, N]
        
        # Extract features
        feat = self.feat_extract(x)  # [B, feat_dim, N]
        
        # Global pooling
        feat_global = feat.max(dim=-1)[0]  # [B, feat_dim]
        
        # Predict realness
        score = self.global_feat(feat_global)  # [B, 1]
        
        return score


class MultiGPUPointCompletionModel(nn.Module):
    """Complete model optimized for multi-GPU training on A100s"""
    def __init__(self,
                 encoder_scales: List[int] = [512, 1024, 2048],
                 encoder_hidden: List[int] = [128, 256, 512],
                 encoder_out: int = 512,
                 transformer_dim: int = 512,
                 transformer_heads: int = 16,  # More heads for A100
                 transformer_layers: int = 8,   # Deeper transformer
                 decoder_stages: int = 5,
                 use_discriminator: bool = False):  # Optional GAN loss
        super().__init__()
        
        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(
            in_dim=6,
            hidden_dims=encoder_hidden,
            out_dim=encoder_out,
            scales=encoder_scales
        )
        
        # Deeper transformer with more heads
        from enhanced_model import GeomMultiTokenTransformer
        self.transformer = GeomMultiTokenTransformer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers
        )
        
        # Feature processing
        self.feat_processor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GroupNorm(32, transformer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # Adaptive decoder
        self.decoder = AdaptiveDecoder(
            feat_dim=transformer_dim,
            num_stages=decoder_stages
        )
        
        # Optional discriminator for adversarial training
        self.use_discriminator = use_discriminator
        if use_discriminator:
            self.discriminator = DiscriminatorNetwork(
                num_points=8192,
                feat_dim=transformer_dim
            )
    
    def forward(self, partial_6d: torch.Tensor, 
                return_discriminator: bool = False) -> torch.Tensor:
        """
        Args:
            partial_6d: [B, N, 6] partial point cloud with normals
            return_discriminator: Whether to return discriminator scores
        Returns:
            completed: [B, 3, 8192] or tuple with discriminator scores
        """
        B, N, _ = partial_6d.shape
        partial_xyz = partial_6d[..., :3]
        
        # Multi-scale encoding
        encoded = self.encoder(partial_6d)  # [B, N, encoder_out]
        
        # Transformer processing with positional bias
        transformed = self.transformer(encoded, partial_xyz)  # [B, N, transformer_dim]
        
        # Process features
        feat_processed = self.feat_processor(transformed)
        
        # Global feature (combine max and mean pooling)
        global_max = feat_processed.max(dim=1)[0]
        global_mean = feat_processed.mean(dim=1)
        global_std = feat_processed.std(dim=1)  # Add std for more info
        global_feat = (global_max + global_mean + global_std) / 3
        
        # Decode to complete point cloud
        completed_xyz = self.decoder(partial_xyz, feat_processed, global_feat)
        
        # Transpose for output format
        completed = completed_xyz.transpose(1, 2)  # [B, 3, 8192]
        
        if return_discriminator and self.use_discriminator:
            disc_score = self.discriminator(completed_xyz)
            return completed, disc_score
        
        return completed


class MultiGPULoss(nn.Module):
    """Enhanced loss function optimized for multi-GPU training"""
    def __init__(self,
                 chamfer_weight: float = 1.0,
                 emd_weight: float = 0.5,  # Can afford EMD with A100s
                 repulsion_weight: float = 0.1,
                 coverage_weight: float = 0.2,
                 smoothness_weight: float = 0.05,
                 gan_weight: float = 0.1,
                 use_emd: bool = True):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.emd_weight = emd_weight if use_emd else 0.0
        self.repulsion_weight = repulsion_weight
        self.coverage_weight = coverage_weight
        self.smoothness_weight = smoothness_weight
        self.gan_weight = gan_weight
        
        if use_emd:
            # Import EMD loss (expensive but better)
            try:
                from emdloss_new import SinkhornEMDLoss
                self.emd_loss = SinkhornEMDLoss(
                    reg=0.01,
                    max_iters=100,
                    num_samples=2048  # Subsample for EMD
                )
            except ImportError:
                print("Warning: EMD loss not available")
                self.emd_loss = None
                self.emd_weight = 0.0
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                partial: torch.Tensor, disc_fake: Optional[torch.Tensor] = None,
                disc_real: Optional[torch.Tensor] = None) -> dict:
        losses = {}
        
        # Chamfer distance
        from minimal_main_4 import chamfer_distance
        losses['chamfer'] = chamfer_distance(pred, gt) * self.chamfer_weight
        
        # EMD loss (if available and enabled)
        if self.emd_weight > 0 and self.emd_loss is not None:
            losses['emd'] = self.emd_loss(pred, gt) * self.emd_weight
        
        # Repulsion loss
        losses['repulsion'] = self.compute_repulsion(pred) * self.repulsion_weight
        
        # Coverage loss
        losses['coverage'] = self.compute_coverage(pred, partial) * self.coverage_weight
        
        # Smoothness loss
        losses['smoothness'] = self.compute_smoothness(pred) * self.smoothness_weight
        
        # GAN loss (if discriminator scores provided)
        if disc_fake is not None and disc_real is not None:
            losses['gan_g'] = F.binary_cross_entropy_with_logits(
                disc_fake, torch.ones_like(disc_fake)
            ) * self.gan_weight
            losses['gan_d'] = (
                F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real)) +
                F.binary_cross_entropy_with_logits(disc_fake.detach(), torch.zeros_like(disc_fake))
            ) * 0.5 * self.gan_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_repulsion(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Enhanced repulsion loss"""
        B, N, _ = points.shape
        
        # Compute k-NN distances
        dist = torch.cdist(points, points)
        dist = dist + torch.eye(N, device=points.device) * 1e6  # Mask self
        knn_dist, _ = dist.topk(k, largest=False, dim=-1)
        
        # Penalize points that are too close
        threshold = 0.005  # Minimum distance threshold
        violation = F.relu(threshold - knn_dist)
        
        return violation.mean()
    
    def compute_coverage(self, pred: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        """Ensure predicted points cover the partial input"""
        # Remove masked points from partial
        B = partial.shape[0]
        coverage_loss = 0.0
        
        for b in range(B):
            mask = partial[b].abs().sum(dim=-1) > 1e-6
            if mask.any():
                partial_valid = partial[b][mask]
                dist = torch.cdist(partial_valid.unsqueeze(0), pred[b:b+1])
                min_dist = dist.min(dim=-1)[0]
                coverage_loss += min_dist.mean()
        
        return coverage_loss / B
    
    def compute_smoothness(self, points: torch.Tensor, k: int = 16) -> torch.Tensor:
        """Local smoothness constraint"""
        B, N, _ = points.shape
        
        # Compute k-NN
        dist = torch.cdist(points, points)
        knn_dist, knn_idx = dist.topk(k, largest=False, dim=-1)
        
        # Compute local covariance
        smooth_loss = 0.0
        for b in range(B):
            for i in range(N):
                neighbors = points[b, knn_idx[b, i]]
                center = neighbors.mean(dim=0, keepdim=True)
                deviations = neighbors - center
                # Penalty is variance (want low variance = smooth)
                smooth_loss += deviations.var()
        
        return smooth_loss / (B * N)


def create_multigpu_model(checkpoint: Optional[str] = None) -> MultiGPUPointCompletionModel:
    """Factory function to create the model"""
    model = MultiGPUPointCompletionModel(
        encoder_scales=[512, 1024, 2048],
        encoder_hidden=[128, 256, 512],
        encoder_out=512,
        transformer_dim=512,
        transformer_heads=16,
        transformer_layers=8,
        decoder_stages=5,
        use_discriminator=False  # Set True if you have memory for GAN training
    )
    
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    return model