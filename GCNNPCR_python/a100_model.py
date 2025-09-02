#!/usr/bin/env python
"""
Multi-GPU optimized model for point cloud completion
Fixed version with architectural improvements to prevent mode collapse
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
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
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


class FoldingBasedDecoder(nn.Module):
    """Folding-based decoder that deforms 2D grids into 3D space"""
    def __init__(self, feat_dim: int = 512, num_seeds: int = 64, grid_size: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_seeds = num_seeds
        self.grid_size = grid_size
        self.points_per_seed = grid_size * grid_size
        
        # Create 2D grid template
        x = torch.linspace(-0.1, 0.1, grid_size)
        y = torch.linspace(-0.1, 0.1, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1))
        
        # Folding network with spatial conditioning
        self.folding_net = nn.Sequential(
            nn.Linear(feat_dim + 2 + 3, 512),  # feat + 2D grid + 3D seed position
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # Output 3D position
        )
    
    def forward(self, seed_xyz: torch.Tensor, seed_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seed_xyz: [B, num_seeds, 3] seed positions
            seed_feat: [B, num_seeds, feat_dim] seed features
        Returns:
            points: [B, num_seeds * points_per_seed, 3]
        """
        B, N, _ = seed_xyz.shape
        device = seed_xyz.device
        
        all_points = []
        for i in range(N):
            # Get seed position and feature
            pos = seed_xyz[:, i:i+1, :]  # [B, 1, 3]
            feat = seed_feat[:, i:i+1, :]  # [B, 1, feat_dim]
            
            # Expand for grid points
            pos_exp = pos.expand(B, self.points_per_seed, 3)
            feat_exp = feat.expand(B, self.points_per_seed, self.feat_dim)
            grid_exp = self.grid.unsqueeze(0).expand(B, -1, -1)
            
            # Concatenate all inputs
            folding_input = torch.cat([feat_exp, grid_exp, pos_exp], dim=-1)
            
            # Generate points through folding
            local_points = self.folding_net(folding_input) + pos_exp
            all_points.append(local_points)
        
        return torch.cat(all_points, dim=1)


class ImprovedAdaptiveDecoder(nn.Module):
    """Improved decoder that prevents collapse through better architecture"""
    def __init__(self, feat_dim: int = 512, num_stages: int = 4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_stages = num_stages
        
        # Learnable offset scales for each stage (not diminishing)
        self.offset_scales = nn.Parameter(torch.ones(num_stages) * 0.1)
        
        # Initial folding decoder for seed generation
        self.seed_generator = FoldingBasedDecoder(feat_dim, num_seeds=64, grid_size=4)
        
        # Progressive upsampling stages with spatial conditioning
        points = 64 * 16  # Start with 1024 points from folding
        self.stages = nn.ModuleList()
        self.coord_refiners = nn.ModuleList()
        
        for i in range(num_stages):
            # Each stage doubles the points
            self.stages.append(SpatiallyAwareRefinementStage(
                feat_dim=feat_dim,
                num_points=points,
                next_points=points * 2,
                stage_idx=i
            ))
            
            # Coordinate refinement network
            self.coord_refiners.append(nn.Sequential(
                nn.Linear(feat_dim + 3 + 3, feat_dim),  # feat + xyz + spread_info
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim // 2),
                nn.LayerNorm(feat_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim // 2, 3)
            ))
            
            points *= 2
        
        # Attention-based feature propagation between stages
        self.propagations = nn.ModuleList([
            AttentionPropagation(feat_dim) for _ in range(num_stages + 1)
        ])
        
        # Anti-collapse regularization
        self.min_spread = 0.2
    
    def forward(self, partial_xyz: torch.Tensor, partial_feat: torch.Tensor,
                local_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_xyz: [B, M, 3] partial input coordinates
            partial_feat: [B, M, feat_dim] partial input features
            local_features: [B, K, feat_dim] multiple local features (not single global)
        Returns:
            points: [B, 8192, 3] completed point cloud
        """
        B = local_features.shape[0]
        device = local_features.device
        
        # Sample seed points from partial input if possible
        num_seeds = min(64, partial_xyz.shape[1])
        if partial_xyz.shape[1] >= num_seeds:
            seed_idx = self.fps_sampling(partial_xyz, num_seeds)
            seed_xyz = torch.gather(partial_xyz, 1, seed_idx.unsqueeze(-1).expand(-1, -1, 3))
            seed_feat = torch.gather(partial_feat, 1, seed_idx.unsqueeze(-1).expand(-1, -1, self.feat_dim))
        else:
            # If not enough points, duplicate
            factor = num_seeds // partial_xyz.shape[1] + 1
            seed_xyz = partial_xyz.repeat(1, factor, 1)[:, :num_seeds]
            seed_feat = partial_feat.repeat(1, factor, 1)[:, :num_seeds]
        
        # Generate initial points through folding
        xyz = self.seed_generator(seed_xyz, seed_feat)  # [B, 1024, 3]
        
        # Initialize features for generated points
        feat = local_features.mean(dim=1, keepdim=True).expand(B, xyz.shape[1], -1)
        
        # Propagate features from partial input
        feat = self.propagations[0](partial_xyz, partial_feat, xyz, feat)
        
        # Progressive refinement with anti-collapse
        for i, (stage, refiner, prop) in enumerate(
            zip(self.stages, self.coord_refiners, self.propagations[1:])):
            
            # Compute spread information
            xyz_mean = xyz.mean(dim=1, keepdim=True)
            xyz_std = xyz.std(dim=1, keepdim=True).expand(-1, xyz.shape[1], -1)
            
            # Refine and upsample with spatial conditioning
            xyz_new, feat_new = stage(xyz, feat, local_features, xyz_std)
            
            # Propagate features from partial input
            feat_new = prop(partial_xyz, partial_feat, xyz_new, feat_new)
            
            # Refine coordinates with spatial awareness
            refine_input = torch.cat([
                feat_new,
                xyz_new,
                xyz_new.std(dim=1, keepdim=True).expand(-1, xyz_new.shape[1], -1)
            ], dim=-1)
            
            # Apply learnable offset scale
            offset = refiner(refine_input) * self.offset_scales[i].abs()
            xyz_new = xyz_new + offset
            
            # Apply anti-collapse repulsion
            xyz_new = self.apply_adaptive_repulsion(xyz_new, min_dist=0.01 / (2 ** i))
            
            # Ensure minimum spread
            current_std = xyz_new.std(dim=1).mean()
            if current_std < self.min_spread:
                # Add noise to break collapse
                noise = torch.randn_like(xyz_new) * (self.min_spread - current_std)
                xyz_new = xyz_new + noise
            
            xyz = xyz_new
            feat = feat_new
        
        return xyz
    
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
    
    def apply_adaptive_repulsion(self, xyz: torch.Tensor, min_dist: float = 0.01) -> torch.Tensor:
        """Apply repulsion force to prevent point clustering"""
        B, N, _ = xyz.shape
        
        # Only apply to a subset for efficiency
        if N > 1000:
            sample_idx = torch.randperm(N, device=xyz.device)[:1000]
            xyz_sample = xyz[:, sample_idx]
        else:
            xyz_sample = xyz
            sample_idx = None
        
        # Compute pairwise distances
        dist = torch.cdist(xyz_sample, xyz_sample)
        dist = dist + torch.eye(xyz_sample.shape[1], device=xyz.device) * 1e6
        
        # Find points that are too close
        too_close = (dist < min_dist).float()
        
        if too_close.any():
            # Compute repulsion forces
            diff = xyz_sample.unsqueeze(2) - xyz_sample.unsqueeze(1)
            forces = diff / (dist.unsqueeze(-1) + 1e-8)
            forces = forces * too_close.unsqueeze(-1)
            total_force = forces.sum(dim=2)
            
            # Apply forces
            if sample_idx is not None:
                xyz[:, sample_idx] = xyz_sample + total_force * 0.01
            else:
                xyz = xyz + total_force * 0.01
        
        return xyz


class SpatiallyAwareRefinementStage(nn.Module):
    """Refinement stage with spatial conditioning"""
    def __init__(self, feat_dim: int, num_points: int, next_points: int, stage_idx: int):
        super().__init__()
        self.num_points = num_points
        self.next_points = next_points
        self.stage_idx = stage_idx
        
        # Feature refinement with spatial input
        self.refine = nn.Sequential(
            nn.Linear(feat_dim * 2 + 3 + 3, feat_dim),  # feat + local_feat + xyz + spread
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive splitting network with spatial awareness
        self.split = nn.Sequential(
            nn.Linear(feat_dim * 2 + 3, feat_dim),  # feat + local_feat + xyz
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)  # 3 for offset1, 3 for offset2
        )
        
        # Use constant scale instead of diminishing
        self.offset_scale = 0.05
    
    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, 
                local_features: torch.Tensor, spread_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = xyz.shape
        
        # Sample local features for each point
        if local_features.shape[1] > 1:
            # Use nearest local feature for each point
            dist_to_local = torch.cdist(xyz, local_features[:, :, :3]) if local_features.shape[-1] > 3 else None
            if dist_to_local is not None:
                nearest_idx = dist_to_local.argmin(dim=-1)  # [B, N]
                local_feat = torch.gather(local_features, 1, 
                                        nearest_idx.unsqueeze(-1).expand(-1, -1, local_features.shape[-1]))
            else:
                local_feat = local_features.mean(dim=1, keepdim=True).expand(B, N, -1)
        else:
            local_feat = local_features.expand(B, N, -1)
        
        # Combine features with spatial information
        combined = torch.cat([feat, local_feat, xyz, spread_info], dim=-1)
        
        # Refine features
        feat_refined = self.refine(combined)
        
        # Generate splitting parameters with spatial conditioning
        split_input = torch.cat([feat_refined, local_feat, xyz], dim=-1)
        split_params = self.split(split_input)
        
        # Use constant scale, not diminishing
        offset1 = split_params[..., :3] * self.offset_scale
        offset2 = split_params[..., 3:] * self.offset_scale
        
        # Add angular variation to prevent all points splitting the same way
        angle = torch.rand(B, N, 1, device=xyz.device) * 2 * np.pi
        rotation = torch.cat([torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)], dim=-1)
        offset2 = offset2 + rotation * 0.02
        
        # Create two points from each original point
        xyz1 = xyz + offset1
        xyz2 = xyz + offset2
        xyz_new = torch.cat([xyz1, xyz2], dim=1)  # [B, N*2, 3]
        
        # Duplicate and refine features
        feat_new = feat_refined.repeat(1, 2, 1)  # [B, N*2, feat_dim]
        
        return xyz_new, feat_new


class MultiGPUPointCompletionModel(nn.Module):
    """Complete model optimized for multi-GPU training with anti-collapse architecture"""
    def __init__(self,
                 encoder_scales: List[int] = [512, 1024, 2048],
                 encoder_hidden: List[int] = [128, 256, 512],
                 encoder_out: int = 512,
                 transformer_dim: int = 512,
                 transformer_heads: int = 16,
                 transformer_layers: int = 8,
                 decoder_stages: int = 4,
                 num_local_features: int = 32):
        super().__init__()
        
        self.num_local_features = num_local_features
        
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
            nn.LayerNorm(transformer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # Local feature extractor (instead of single global)
        self.local_feat_extractor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # Improved adaptive decoder
        self.decoder = ImprovedAdaptiveDecoder(
            feat_dim=transformer_dim,
            num_stages=decoder_stages
        )
    
    def forward(self, partial_6d: torch.Tensor, 
                return_discriminator: bool = False) -> torch.Tensor:
        """
        Args:
            partial_6d: [B, N, 6] partial point cloud with normals
            return_discriminator: Whether to return discriminator scores
        Returns:
            completed: [B, 3, 8192] completed point cloud
        """
        B, N, _ = partial_6d.shape
        partial_xyz = partial_6d[..., :3]
        
        # Multi-scale encoding
        encoded = self.encoder(partial_6d)  # [B, N, encoder_out]
        
        # Transformer processing with positional bias
        transformed = self.transformer(encoded, partial_xyz)  # [B, N, transformer_dim]
        
        # Process features
        feat_processed = self.feat_processor(transformed)
        
        # Extract multiple local features instead of single global
        if N >= self.num_local_features:
            # FPS sample for diverse local features
            local_idx = self.fps_sampling(partial_xyz, self.num_local_features)
            local_xyz = torch.gather(partial_xyz, 1, local_idx.unsqueeze(-1).expand(-1, -1, 3))
            local_feat = torch.gather(feat_processed, 1, 
                                     local_idx.unsqueeze(-1).expand(-1, -1, feat_processed.shape[-1]))
        else:
            # Use all points if fewer than num_local_features
            local_xyz = partial_xyz
            local_feat = feat_processed
        
        # Process local features
        local_feat = self.local_feat_extractor(local_feat)
        
        # Combine position and features for decoder
        local_features = torch.cat([local_xyz, local_feat], dim=-1)[:, :, 3:]  # Keep only features
        
        # Decode to complete point cloud
        completed_xyz = self.decoder(partial_xyz, feat_processed, local_features)
        
        # Soft bounding to prevent extreme values
        completed_xyz = completed_xyz / (1 + completed_xyz.abs()) * 2.0  # Softsign scaled to [-2, 2]
        
        # Handle NaN/Inf
        completed_xyz = torch.nan_to_num(
            completed_xyz, nan=0.0, posinf=1.0, neginf=-1.0
        )
        
        # Transpose for output format
        completed = completed_xyz.transpose(1, 2)  # [B, 3, 8192]
        
        return completed
    
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


class EnhancedMultiGPULoss(nn.Module):
    """Enhanced loss function with diversity term to prevent collapse"""
    def __init__(self,
                 chamfer_weight: float = 1.0,
                 emd_weight: float = 0.5,
                 repulsion_weight: float = 0.2,  # Increased
                 coverage_weight: float = 0.2,
                 smoothness_weight: float = 0.05,
                 diversity_weight: float = 0.3,  # New term
                 min_spread: float = 0.3,
                 use_emd: bool = True):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.emd_weight = emd_weight if use_emd else 0.0
        self.repulsion_weight = repulsion_weight
        self.coverage_weight = coverage_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
        self.min_spread = min_spread
        
        if use_emd:
            try:
                from emdloss_new import SinkhornEMDLoss
                self.emd_loss = SinkhornEMDLoss(
                    reg=0.01,
                    max_iters=100,
                    num_samples=2048
                )
            except ImportError:
                print("Warning: EMD loss not available")
                self.emd_loss = None
                self.emd_weight = 0.0
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, 
                partial: torch.Tensor, disc_fake: Optional[torch.Tensor] = None,
                disc_real: Optional[torch.Tensor] = None) -> dict:
        losses = {}
        
        # Fix dimensions if needed
        pred_points = pred.transpose(1, 2) if pred.shape[1] == 3 else pred
        
        # Chamfer distance
        from minimal_main_4 import chamfer_distance
        losses['chamfer'] = chamfer_distance(pred_points, gt) * self.chamfer_weight
        
        # EMD loss
        if self.emd_weight > 0 and self.emd_loss is not None:
            with torch.cuda.amp.autocast(enabled=False):
                emd_val = self.emd_loss(pred_points, gt)
            losses['emd'] = emd_val * self.emd_weight
        
        # Enhanced repulsion loss
        losses['repulsion'] = self.compute_enhanced_repulsion(pred_points) * self.repulsion_weight
        
        # Coverage loss
        losses['coverage'] = self.compute_coverage(pred_points, partial) * self.coverage_weight
        
        # Smoothness loss
        losses['smoothness'] = self.compute_smoothness(pred_points) * self.smoothness_weight
        
        # Diversity loss (anti-collapse)
        losses['diversity'] = self.compute_diversity_loss(pred_points) * self.diversity_weight
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_enhanced_repulsion(self, points: torch.Tensor, k: int = 8) -> torch.Tensor:
        """Stronger repulsion with multiple distance thresholds"""
        B, N, _ = points.shape
        
        # Compute k-NN distances
        dist = torch.cdist(points, points)
        dist = dist + torch.eye(N, device=points.device) * 1e6
        knn_dist, _ = dist.topk(k, largest=False, dim=-1)
        
        # Multiple thresholds for different scales
        loss = 0.0
        thresholds = [0.005, 0.01, 0.02]
        weights = [10.0, 5.0, 1.0]
        
        for threshold, weight in zip(thresholds, weights):
            violation = F.relu(threshold - knn_dist)
            loss += violation.mean() * weight
        
        return loss
    
    def compute_diversity_loss(self, points: torch.Tensor) -> torch.Tensor:
        """Penalize low diversity in point distribution"""
        B, N, _ = points.shape
        
        # Compute per-batch statistics
        std = points.std(dim=1)  # [B, 3]
        mean_std = std.mean(dim=-1)  # [B]
        
        # Penalize if spread is too small
        diversity_penalty = F.relu(self.min_spread - mean_std).mean()
        
        # Also penalize if all points are too similar (low variance of distances)
        for b in range(B):
            pairwise_dist = torch.cdist(points[b:b+1], points[b:b+1])[0]
            dist_std = pairwise_dist[pairwise_dist > 0].std()
            diversity_penalty += F.relu(0.1 - dist_std)
        
        return diversity_penalty / B
    
    def compute_coverage(self, pred: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        """Ensure predicted points cover the partial input"""
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
            for i in range(min(N, 500)):  # Sample for efficiency
                neighbors = points[b, knn_idx[b, i]]
                center = neighbors.mean(dim=0, keepdim=True)
                deviations = neighbors - center
                smooth_loss += deviations.var()
        
        return smooth_loss / (B * min(N, 500))


def create_multigpu_model(checkpoint: Optional[str] = None) -> MultiGPUPointCompletionModel:
    """Factory function to create the improved model"""
    model = MultiGPUPointCompletionModel(
        encoder_scales=[512, 1024, 2048],
        encoder_hidden=[128, 256, 512],
        encoder_out=512,
        transformer_dim=512,
        transformer_heads=16,
        transformer_layers=8,
        decoder_stages=4,  # Reduced from 5 for 8192 points
        num_local_features=32
    )
    
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    return model