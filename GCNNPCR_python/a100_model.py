#!/usr/bin/env python
"""
Redesigned point cloud completion model based on proven architectures
Incorporates principles from PCN, FoldingNet, SnowflakeNet, and PoinTr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import math

# Import base components
from minimal_main_4 import GraphEncoder


class GridGenerator(nn.Module):
    """Generate structured 2D/3D grids as base structures"""
    
    @staticmethod
    def square_grid(size: int) -> torch.Tensor:
        """Generate 2D square grid"""
        points = []
        for i in range(size):
            for j in range(size):
                x = (i / (size - 1) - 0.5) * 2  # Range [-1, 1]
                y = (j / (size - 1) - 0.5) * 2
                points.append([x, y])
        return torch.tensor(points, dtype=torch.float32)
    
    @staticmethod
    def sphere_grid(size: int) -> torch.Tensor:
        """Generate points on unit sphere using Fibonacci spiral"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        
        for i in range(size):
            y = 1 - (i / float(size - 1)) * 2  # -1 to 1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        
        return torch.tensor(points, dtype=torch.float32)
    
    @staticmethod
    def cylinder_grid(height_samples: int, radial_samples: int) -> torch.Tensor:
        """Generate points on cylinder surface"""
        points = []
        for h in range(height_samples):
            height = (h / (height_samples - 1) - 0.5) * 2
            for r in range(radial_samples):
                angle = 2 * np.pi * r / radial_samples
                x = np.cos(angle)
                z = np.sin(angle)
                points.append([x, height, z])
        return torch.tensor(points, dtype=torch.float32)


class PCNEncoder(nn.Module):
    """Encoder inspired by Point Completion Network with LayerNorm."""
    def __init__(self, in_dim: int = 6, out_dim: int = 1024):
        super().__init__()

        # PointNet-style encoder with skip connections
        self.conv1 = nn.Conv1d(in_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, out_dim, 1)

        # Use LayerNorm for better stability with variable batch sizes
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(out_dim)

        # Feature aggregation
        self.final_conv = nn.Conv1d(128 + 256 + 512 + out_dim, out_dim, 1)
        self.final_ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, N, 6] input points with normals
        Returns:
            global_feat: [B, out_dim] global feature
            point_feats: List of intermediate features for skip connections
        """
        B, N, C = x.shape
        
        # Transpose for Conv1d
        x = x.transpose(1, 2)  # [B, 6, N]

        # Extract hierarchical features with LayerNorm and ReLU
        feat1 = self.conv1(x).transpose(1, 2)  # [B, N, 128]
        feat1 = F.relu(self.ln1(feat1)).transpose(1, 2)  # [B, 128, N]
        
        feat2 = self.conv2(feat1).transpose(1, 2)  # [B, N, 256]
        feat2 = F.relu(self.ln2(feat2)).transpose(1, 2)  # [B, 256, N]
        
        feat3 = self.conv3(feat2).transpose(1, 2)  # [B, N, 512]
        feat3 = F.relu(self.ln3(feat3)).transpose(1, 2)  # [B, 512, N]
        
        feat4 = self.conv4(feat3).transpose(1, 2)  # [B, N, out_dim]
        feat4 = F.relu(self.ln4(feat4)).transpose(1, 2)  # [B, out_dim, N]

        # Concatenate multi-scale features
        concat_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        combined = self.final_conv(concat_feat).transpose(1, 2)  # [B, N, out_dim]
        combined = F.relu(self.final_ln(combined)).transpose(1, 2)  # [B, out_dim, N]

        # Global feature via max pooling
        global_feat = combined.max(dim=2)[0]  # [B, out_dim]

        # Store point-wise features for skip connections
        point_feats = [
            feat1.transpose(1, 2),  # [B, N, 128]
            feat2.transpose(1, 2),  # [B, N, 256]
            feat3.transpose(1, 2),  # [B, N, 512]
            feat4.transpose(1, 2),  # [B, N, out_dim]
        ]

        return global_feat, point_feats


class FoldingDecoder(nn.Module):
    """Folding-based decoder with dropout and LayerNorm."""
    def __init__(self, feat_dim: int = 1024, num_patches: int = 4, points_per_patch: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.points_per_patch = points_per_patch
        self.grid_size = int(np.sqrt(points_per_patch))

        # Generate base 2D grid
        self.register_buffer('base_grid', GridGenerator.square_grid(self.grid_size))

        # Multiple folding networks
        self.folding_nets = nn.ModuleList()
        for _ in range(num_patches):
            folding = nn.Sequential(
                nn.Linear(feat_dim + 2, 512),
                nn.ReLU(),
                nn.LayerNorm(512),  # LayerNorm instead of InstanceNorm
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),  # LayerNorm instead of InstanceNorm
                nn.Dropout(0.1),
                nn.Linear(256, 3),
            )
            self.folding_nets.append(folding)

        # Second-stage folding for refinement
        self.refine_net = nn.Sequential(
            nn.Linear(feat_dim + 3, 512),
            nn.ReLU(),
            nn.LayerNorm(512),  # LayerNorm instead of InstanceNorm
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # LayerNorm instead of InstanceNorm
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: [B, feat_dim]
        Returns:
            points: [B, num_patches * points_per_patch, 3]
        """
        B = global_feat.shape[0]
        all_points = []

        for folding_net in self.folding_nets:
            # Expand features for all grid points
            feat_expanded = global_feat.unsqueeze(1).expand(B, self.points_per_patch, -1)
            grid_expanded = self.base_grid.unsqueeze(0).expand(B, -1, -1).to(global_feat.device)

            # Reshape for processing
            feat_expanded_flat = feat_expanded.reshape(B * self.points_per_patch, -1)
            grid_expanded_flat = grid_expanded.reshape(B * self.points_per_patch, -1)

            # First folding
            folding_input = torch.cat([feat_expanded_flat, grid_expanded_flat], dim=-1)
            folded = folding_net(folding_input)
            folded = folded.reshape(B, self.points_per_patch, 3)

            # Second folding for refinement
            refine_input = torch.cat([feat_expanded_flat, folded.reshape(B * self.points_per_patch, 3)], dim=-1)
            refined = self.refine_net(refine_input)
            refined = refined.reshape(B, self.points_per_patch, 3)

            # Add residual
            final_points = folded + refined * 0.1
            all_points.append(final_points)

        return torch.cat(all_points, dim=1)


class SnowflakeDecoder(nn.Module):
    """Snowflake-style hierarchical decoder with skip connections"""
    def __init__(self, feat_dim: int = 1024, coarse_points: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.coarse_points = coarse_points
        
        # Coarse point generation using folding
        self.coarse_generator = FoldingDecoder(feat_dim, num_patches=2, points_per_patch=256)
        
        # Skip-attention modules for each level
        self.skip_attentions = nn.ModuleList([
            SkipAttention(feat_dim, 512),  # 512 -> 1024
            SkipAttention(feat_dim, 1024), # 1024 -> 2048
            SkipAttention(feat_dim, 2048), # 2048 -> 4096
            SkipAttention(feat_dim, 4096), # 4096 -> 8192
        ])
        
        # Child point generators (split each point into 2)
        self.child_generators = nn.ModuleList([
            ChildPointGenerator(feat_dim),
            ChildPointGenerator(feat_dim),
            ChildPointGenerator(feat_dim),
            ChildPointGenerator(feat_dim),
        ])
    
    def forward(self, global_feat: torch.Tensor, partial_xyz: torch.Tensor, 
                partial_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: [B, feat_dim]
            partial_xyz: [B, M, 3] partial input coordinates
            partial_feat: [B, M, feat_dim] partial input features
        Returns:
            points: [B, 8192, 3]
        """
        B = global_feat.shape[0]
        
        # Generate coarse points
        coarse_xyz = self.coarse_generator(global_feat)  # [B, 512, 3]
        
        # Initialize features for coarse points
        coarse_feat = global_feat.unsqueeze(1).expand(B, self.coarse_points, -1)
        
        # Hierarchical refinement with skip connections
        xyz = coarse_xyz
        feat = coarse_feat
        
        for skip_attn, child_gen in zip(self.skip_attentions, self.child_generators):
            # Apply skip attention to get features from partial input
            feat_with_skip = skip_attn(xyz, feat, partial_xyz, partial_feat)
            
            # Generate child points
            xyz, feat = child_gen(xyz, feat_with_skip, global_feat)
        
        return xyz


class SkipAttention(nn.Module):
    """Cross-attention from generated points to partial input with dropout."""
    def __init__(self, feat_dim: int, num_points: int):
        super().__init__()
        self.num_points = num_points

        # Query, Key, Value projections
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)

        # Output projection with dropout for stability
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(0.1)          # dropout layer
        )

        # LayerNorm instead of BatchNorm for better small-batch behaviour
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, query_xyz: torch.Tensor, query_feat: torch.Tensor,
                key_xyz: torch.Tensor, key_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_xyz: [B, N, 3] generated point positions
            query_feat: [B, N, feat_dim] generated point features
            key_xyz: [B, M, 3] partial input positions
            key_feat: [B, M, feat_dim] partial input features
        Returns:
            updated_feat: [B, N, feat_dim]
        """
        B, N, _ = query_xyz.shape

        # Project to query/key/value
        Q = self.q_proj(query_feat)  # [B, N, feat_dim]
        K = self.k_proj(key_feat)    # [B, M, feat_dim]
        V = self.v_proj(key_feat)    # [B, M, feat_dim]

        # Compute attention scores with positional bias
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
        # Gaussian distance weighting
        dist = torch.cdist(query_xyz, key_xyz)  # [B, N, M]
        dist_weight = torch.exp(-dist * 2.0)
        scores = scores + torch.log(dist_weight + 1e-8)

        # Softmax and dropout on attention map
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        # Aggregate values
        attended = torch.bmm(attn, V)

        # Residual connection with layer normalisation and dropout on the output projection
        output = self.norm(query_feat + self.out_proj(attended))

        return output


class ChildPointGenerator(nn.Module):
    """Generate child points from parent points with controlled split scale"""
    def __init__(self, feat_dim: int):
        super().__init__()

        # Network to generate splitting parameters
        self.split_net = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),          # dropout for stability
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),          # dropout for stability
            nn.Linear(128, 6)         # 2 children x 3 coordinates
        )

        # Feature refinement for children
        self.feat_refine = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),          # dropout for stability
            nn.Linear(feat_dim, feat_dim)
        )

        # Learnable splitting scale (initialised small)
        self.split_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, parent_xyz: torch.Tensor, parent_feat: torch.Tensor,
                global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            parent_xyz: [B, N, 3]
            parent_feat: [B, N, feat_dim]
            global_feat: [B, feat_dim]
        Returns:
            child_xyz: [B, N*2, 3]
            child_feat: [B, N*2, feat_dim]
        """
        B, N, _ = parent_xyz.shape

        # Combine with global features
        global_expanded = global_feat.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([parent_feat, global_expanded], dim=-1)

        # Generate child offsets
        offsets = self.split_net(combined)  # [B, N, 6]

        # Clamp the splitting scale to prevent runaway growth
        scale = torch.clamp(self.split_scale.abs(), max=0.05)

        offset1 = offsets[..., :3] * scale
        offset2 = offsets[..., 3:] * scale

        # Generate child points
        child1_xyz = parent_xyz + offset1
        child2_xyz = parent_xyz + offset2
        child_xyz = torch.cat([child1_xyz, child2_xyz], dim=1)

        # Refine features for children
        child_feat = self.feat_refine(combined)
        child_feat = torch.cat([child_feat, child_feat], dim=1)

        return child_xyz, child_feat


class AntiCollapsePointCompletion(nn.Module):
    """Main model with multiple anti-collapse mechanisms"""
    def __init__(self, encoder_type: str = 'pcn', decoder_type: str = 'snowflake'):
        super().__init__()
        
        # Encoder
        self.encoder_type = encoder_type
        if encoder_type == 'pcn':
            self.encoder = PCNEncoder(in_dim=6, out_dim=1024)
        else:
            # Use graph encoder from original
            self.encoder = GraphEncoder(in_dim=6, hidden_dims=[128, 256, 512], out_dim=1024, k=16)
        
        # Feature processor with positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)
        )
        
        # Decoder
        self.decoder_type = decoder_type
        if decoder_type == 'snowflake':
            self.decoder = SnowflakeDecoder(feat_dim=1024, coarse_points=512)
        else:
            self.decoder = FoldingDecoder(feat_dim=1024, num_patches=8, points_per_patch=1024)
        
        # Output regularization
        self.output_norm = nn.LayerNorm(3)
    
    def forward(self, partial_6d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_6d: [B, N, 6] partial point cloud with normals
        Returns:
            completed: [B, 3, 8192]
        """
        B, N, _ = partial_6d.shape
        partial_xyz = partial_6d[..., :3]
        
        # Encode
        if self.encoder_type == 'pcn':
            global_feat, point_feats = self.encoder(partial_6d)
            # Use the last point features
            partial_feat = point_feats[-1]  # [B, N, 1024]
        else:
            partial_feat = self.encoder(partial_6d)  # [B, N, 1024]
            global_feat = partial_feat.max(dim=1)[0]  # [B, 1024]
        
        # Add positional encoding to global feature
        pos_feat = self.pos_encoder(partial_xyz.mean(dim=1))  # [B, 1024]
        global_feat = global_feat + pos_feat * 0.1
        
        # Decode
        if self.decoder_type == 'snowflake':
            completed_xyz = self.decoder(global_feat, partial_xyz, partial_feat)
        else:
            completed_xyz = self.decoder(global_feat)
        
        # Normalize output to prevent unbounded growth
        # Normalise each point's coordinates using LayerNorm
        completed_xyz = self.output_norm(completed_xyz)
        
        # Apply tanh for final bounding
        completed_xyz = torch.tanh(completed_xyz) * 1.5  # Scale to [-1.5, 1.5]
        
        # Add connection to partial input (residual-style)
        # Sample partial points to match output size if needed
        if N >= completed_xyz.shape[1]:
            indices = torch.randperm(N, device=partial_xyz.device)[:completed_xyz.shape[1]]
            partial_sample = partial_xyz[:, indices]
        else:
            # Repeat partial points
            factor = completed_xyz.shape[1] // N + 1
            partial_sample = partial_xyz.repeat(1, factor, 1)[:, :completed_xyz.shape[1]]
        
        # Blend with partial input to maintain structure
        alpha = 0.1  # Blending factor
        completed_xyz = completed_xyz * (1 - alpha) + partial_sample * alpha
        
        # Transpose for output format
        completed = completed_xyz.transpose(1, 2)  # [B, 3, 8192]
        
        return completed


class ImprovedLoss(nn.Module):
    """Loss function with strong anti-collapse terms"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, partial: torch.Tensor) -> dict:
        """
        Args:
            pred: [B, N, 3] or [B, 3, N]
            gt: [B, M, 3]
            partial: [B, K, 3]
        """
        # Handle dimension
        if pred.shape[1] == 3:
            pred = pred.transpose(1, 2)
        
        losses = {}
        
        # 1. Chamfer Distance
        from minimal_main_4 import chamfer_distance
        losses['chamfer'] = chamfer_distance(pred, gt)
        
        # 2. Earth Mover's Distance approximation
        if pred.shape[1] == gt.shape[1]:
            # Simple L2 after sorting
            pred_sorted = torch.sort(pred.view(-1, 3), dim=0)[0]
            gt_sorted = torch.sort(gt.view(-1, 3), dim=0)[0]
            losses['emd'] = self.mse(pred_sorted, gt_sorted) * 0.1
        else:
            losses['emd'] = torch.tensor(0.0).to(pred.device)
        
        # 3. Coverage loss (must cover partial input)
        dist_to_partial = torch.cdist(partial, pred)
        coverage = dist_to_partial.min(dim=-1)[0].mean()
        losses['coverage'] = coverage * 5.0  # Strong weight
        
        # 4. Uniform distribution loss
        # Points should be uniformly distributed
        pred_flat = pred.reshape(-1, 3)
        K = min(50, pred_flat.shape[0])
        if K > 1:
            # Compute K-NN distances
            dist = torch.cdist(pred_flat.unsqueeze(0), pred_flat.unsqueeze(0))[0]
            knn_dist, _ = dist.topk(K, largest=False, dim=-1)
            # Variance of K-NN distances should be low (uniform spacing)
            uniformity = knn_dist[:, 1:].std(dim=-1).mean()
            losses['uniformity'] = uniformity * 2.0
        else:
            losses['uniformity'] = torch.tensor(0.0).to(pred.device)
        
        # 5. Spread loss (anti-collapse)
        std = pred.std(dim=1).mean()
        target_std = 0.5  # Target standard deviation
        losses['spread'] = F.relu(target_std - std) * 10.0  # Very strong weight
        
        # 6. Repulsion loss
        min_dist = 0.01
        if pred.shape[1] > 100:
            # Sample for efficiency
            sample_idx = torch.randperm(pred.shape[1], device=pred.device)[:100]
            pred_sample = pred[:, sample_idx]
        else:
            pred_sample = pred
        
        dist = torch.cdist(pred_sample, pred_sample)
        dist = dist + torch.eye(pred_sample.shape[1], device=dist.device) * 1e10
        violations = F.relu(min_dist - dist)
        losses['repulsion'] = violations.mean() * 5.0
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


def create_anticollapsemodel(checkpoint: Optional[str] = None) -> AntiCollapsePointCompletion:
    """Factory function"""
    model = AntiCollapsePointCompletion(encoder_type='pcn', decoder_type='snowflake')
    
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    return model