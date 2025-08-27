#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced model architecture for better point cloud completion
Addresses issues with the original model that cause "center blob" outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

# Import base components from minimal_main_4
from minimal_main_4 import (
    GraphEncoder,
    GeomMultiTokenTransformer,
    local_knn,
    fps_subsample,
    MLP_Res
)


class FoldingDecoder(nn.Module):
    """
    Folding-based decoder that deforms a 2D grid into 3D space
    This provides better spatial structure than random seed points
    """
    def __init__(self, feat_dim: int = 128, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points
        self.grid_size = int(np.sqrt(num_points))
        
        # Create 2D grid
        x = torch.linspace(-1, 1, self.grid_size)
        y = torch.linspace(-1, 1, self.grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        self.grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_points, 2]
        
        # Folding network
        self.fold1 = nn.Sequential(
            nn.Linear(feat_dim + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        self.fold2 = nn.Sequential(
            nn.Linear(feat_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_feat: [B, feat_dim]
        Returns:
            points: [B, num_points, 3]
        """
        B = global_feat.shape[0]
        grid = self.grid.to(global_feat.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, 2]
        
        # Expand features
        feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_points, -1)  # [B, N, feat_dim]
        
        # First folding
        fold1_input = torch.cat([feat_expanded, grid], dim=-1)
        fold1_out = self.fold1(fold1_input)  # [B, N, 3]
        
        # Second folding
        fold2_input = torch.cat([feat_expanded, fold1_out], dim=-1)
        points = self.fold2(fold2_input)  # [B, N, 3]
        
        return points


class AttentionPropagation(nn.Module):
    """
    Propagate features from sparse to dense points using attention
    """
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(feat_dim + 3, feat_dim)
        self.k_proj = nn.Linear(feat_dim + 3, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5
    
    def forward(self, sparse_xyz: torch.Tensor, sparse_feat: torch.Tensor,
                dense_xyz: torch.Tensor, dense_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_xyz: [B, N1, 3] positions of sparse points
            sparse_feat: [B, N1, feat_dim] features of sparse points
            dense_xyz: [B, N2, 3] positions of dense points
            dense_feat: [B, N2, feat_dim] features of dense points
        Returns:
            propagated_feat: [B, N2, feat_dim]
        """
        B, N1, _ = sparse_xyz.shape
        B, N2, _ = dense_xyz.shape
        
        # Compute queries from dense points
        q_input = torch.cat([dense_xyz, dense_feat], dim=-1)
        Q = self.q_proj(q_input)  # [B, N2, feat_dim]
        
        # Compute keys and values from sparse points
        k_input = torch.cat([sparse_xyz, sparse_feat], dim=-1)
        K = self.k_proj(k_input)  # [B, N1, feat_dim]
        V = self.v_proj(sparse_feat)  # [B, N1, feat_dim]
        
        # Attention
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, N2, N1]
        
        # Add distance penalty to attention scores
        dist = torch.cdist(dense_xyz, sparse_xyz)  # [B, N2, N1]
        dist_penalty = -dist * 0.5  # Nearby points get higher attention
        scores = scores + dist_penalty
        
        attn = F.softmax(scores, dim=-1)
        
        # Propagate features
        propagated = torch.bmm(attn, V)  # [B, N2, feat_dim]
        output = self.out_proj(propagated + dense_feat)
        
        return output


class HierarchicalRefinementDecoder(nn.Module):
    """
    Hierarchical decoder with skip connections and attention-based refinement
    """
    def __init__(self, feat_dim: int = 128, coarse_points: int = 512):
        super().__init__()
        self.coarse_points = coarse_points
        
        # Initial coarse generation (using folding)
        self.coarse_decoder = FoldingDecoder(feat_dim, coarse_points)
        
        # Hierarchical refinement stages
        self.stages = nn.ModuleList([
            RefinementStage(feat_dim, scale_factor=2),  # 512 -> 1024
            RefinementStage(feat_dim, scale_factor=2),  # 1024 -> 2048
            RefinementStage(feat_dim, scale_factor=2),  # 2048 -> 4096
            RefinementStage(feat_dim, scale_factor=2),  # 4096 -> 8192
        ])
        
        # Feature propagation between stages
        self.propagations = nn.ModuleList([
            AttentionPropagation(feat_dim) for _ in range(len(self.stages))
        ])
    
    def forward(self, partial_xyz: torch.Tensor, partial_feat: torch.Tensor,
                global_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_xyz: [B, M, 3] partial input coordinates
            partial_feat: [B, M, feat_dim] partial input features
            global_feat: [B, feat_dim] global features
        Returns:
            points: [B, 8192, 3] completed point cloud
        """
        B = global_feat.shape[0]
        
        # Generate initial coarse points
        coarse_xyz = self.coarse_decoder(global_feat)  # [B, 512, 3]
        coarse_feat = global_feat.unsqueeze(1).expand(-1, self.coarse_points, -1)
        
        # Hierarchical refinement
        curr_xyz = coarse_xyz
        curr_feat = coarse_feat
        
        for stage, prop in zip(self.stages, self.propagations):
            # Upsample points
            next_xyz, next_feat = stage(curr_xyz, curr_feat, global_feat)
            
            # Propagate features from partial input
            next_feat = prop(partial_xyz, partial_feat, next_xyz, next_feat)
            
            curr_xyz = next_xyz
            curr_feat = next_feat
        
        return curr_xyz


class RefinementStage(nn.Module):
    """
    Single refinement stage that upsamples points
    """
    def __init__(self, feat_dim: int = 128, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        self.offset_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 3 * scale_factor)
        )
        
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim * scale_factor)
        )
    
    def forward(self, xyz: torch.Tensor, feat: torch.Tensor, 
                global_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, N, 3] input points
            feat: [B, N, feat_dim] input features
            global_feat: [B, feat_dim] global features
        Returns:
            new_xyz: [B, N*scale_factor, 3]
            new_feat: [B, N*scale_factor, feat_dim]
        """
        B, N, _ = xyz.shape
        
        # Combine with global features
        global_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        combined_feat = torch.cat([feat, global_expanded], dim=-1)
        
        # Generate offsets
        offsets = self.offset_net(combined_feat)  # [B, N, 3*scale_factor]
        offsets = offsets.view(B, N, self.scale_factor, 3)
        offsets = offsets * 0.1  # Scale down offsets
        
        # Generate new features
        new_feat = self.feat_net(combined_feat)  # [B, N, feat_dim*scale_factor]
        new_feat = new_feat.view(B, N * self.scale_factor, -1)
        
        # Create new points
        xyz_expanded = xyz.unsqueeze(2).expand(-1, -1, self.scale_factor, -1)  # [B, N, scale_factor, 3]
        new_xyz = xyz_expanded + offsets
        new_xyz = new_xyz.view(B, N * self.scale_factor, 3)
        
        return new_xyz, new_feat


class EnhancedPointCompletionModel(nn.Module):
    """
    Enhanced model with better architecture for point cloud completion
    """
    def __init__(self,
                 encoder_hidden_dims: List[int] = [128, 256],
                 encoder_out_dim: int = 256,
                 transformer_dim: int = 256,
                 transformer_heads: int = 8,
                 transformer_layers: int = 6,
                 coarse_points: int = 512,
                 use_attention_encoder: bool = True):
        super().__init__()
        
        # Enhanced encoder with skip connections
        self.encoder = GraphEncoder(
            in_dim=6,
            hidden_dims=encoder_hidden_dims,
            out_dim=encoder_out_dim,
            k=16,
            use_attention=use_attention_encoder
        )
        
        # Deeper transformer for better feature extraction
        self.transformer = GeomMultiTokenTransformer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers
        )
        
        # Bridge if dimensions don't match
        self.bridge = nn.Linear(encoder_out_dim, transformer_dim) if encoder_out_dim != transformer_dim else nn.Identity()
        
        # Feature extraction from partial input
        self.partial_feat_extractor = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        # Enhanced hierarchical decoder
        self.decoder = HierarchicalRefinementDecoder(
            feat_dim=transformer_dim,
            coarse_points=coarse_points
        )
        
        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
    
    def forward(self, partial_6d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_6d: [B, N, 6] partial point cloud with normals
        Returns:
            completed: [B, 3, 8192] completed point cloud
        """
        B, N, _ = partial_6d.shape
        
        # Extract coordinates and features
        partial_xyz = partial_6d[..., :3]
        
        # Encode partial input
        encoded = self.encoder(partial_6d)  # [B, N, encoder_out_dim]
        
        # Map to transformer dimension
        features = self.bridge(encoded)  # [B, N, transformer_dim]
        
        # Apply transformer with geometry bias
        features = self.transformer(features, partial_xyz)  # [B, N, transformer_dim]
        
        # Extract per-point features
        partial_feat = self.partial_feat_extractor(features)  # [B, N, transformer_dim]
        
        # Global feature aggregation (using both max and mean)
        global_max = features.max(dim=1)[0]  # [B, transformer_dim]
        global_mean = features.mean(dim=1)   # [B, transformer_dim]
        global_feat = (global_max + global_mean) / 2  # [B, transformer_dim]
        
        # Decode to complete point cloud
        completed_xyz = self.decoder(partial_xyz, partial_feat, global_feat)  # [B, 8192, 3]
        
        # Final refinement
        completed = completed_xyz.transpose(1, 2)  # [B, 3, 8192]
        completed = self.final_refine(completed) + completed  # Residual connection
        
        return completed


class CombinedLossWithEMD(nn.Module):
    """
    Enhanced loss function with multiple terms
    """
    def __init__(self, 
                 chamfer_weight: float = 1.0,
                 repulsion_weight: float = 0.01,
                 smoothness_weight: float = 0.005,
                 coverage_weight: float = 0.1):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.repulsion_weight = repulsion_weight
        self.smoothness_weight = smoothness_weight
        self.coverage_weight = coverage_weight
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor, partial: torch.Tensor) -> dict:
        """
        Args:
            pred: [B, N, 3] predicted points
            gt: [B, M, 3] ground truth points
            partial: [B, K, 3] partial input points
        Returns:
            Dictionary with total loss and individual components
        """
        from minimal_main_4 import chamfer_distance, repulsion_loss
        
        # Chamfer distance
        cd_loss = chamfer_distance(pred, gt)
        
        # Repulsion loss (prevent clustering)
        rep_loss = repulsion_loss(pred, k=4, threshold=0.01)
        
        # Smoothness loss (local coherence)
        smooth_loss = self.compute_smoothness_loss(pred)
        
        # Coverage loss (ensure we cover the partial input)
        coverage_loss = self.compute_coverage_loss(pred, partial)
        
        # Total loss
        total = (self.chamfer_weight * cd_loss +
                self.repulsion_weight * rep_loss +
                self.smoothness_weight * smooth_loss +
                self.coverage_weight * coverage_loss)
        
        return {
            'total': total,
            'chamfer': cd_loss,
            'repulsion': rep_loss,
            'smoothness': smooth_loss,
            'coverage': coverage_loss
        }
    
    def compute_smoothness_loss(self, points: torch.Tensor) -> torch.Tensor:
        """Encourage local smoothness"""
        B, N, _ = points.shape
        
        # Find k nearest neighbors
        k = min(16, N)
        dist = torch.cdist(points, points)
        knn_dist, _ = dist.topk(k, largest=False, dim=-1)
        
        # Variance of distances to neighbors (lower is smoother)
        smooth_loss = knn_dist.var(dim=-1).mean()
        
        return smooth_loss
    
    def compute_coverage_loss(self, pred: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        """Ensure predicted points cover the partial input"""
        # For each partial point, find distance to nearest predicted point
        dist = torch.cdist(partial, pred)  # [B, K, N]
        min_dist = dist.min(dim=-1)[0]  # [B, K]
        
        # Penalize if partial points are far from any predicted point
        coverage_loss = min_dist.mean()
        
        return coverage_loss


def create_enhanced_model(checkpoint_path: Optional[str] = None) -> EnhancedPointCompletionModel:
    """
    Create the enhanced model, optionally loading from checkpoint
    """
    model = EnhancedPointCompletionModel(
        encoder_hidden_dims=[128, 256, 256],
        encoder_out_dim=256,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=6,
        coarse_points=512,
        use_attention_encoder=True
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            model.load_state_dict(checkpoint)
    
    return model


if __name__ == "__main__":
    # Test the model
    model = create_enhanced_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    num_points = 4096
    partial = torch.randn(batch_size, num_points, 6)
    
    with torch.no_grad():
        output = model(partial)
    
    print(f"Input shape: {partial.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test loss
    loss_fn = CombinedLossWithEMD()
    pred = output.permute(0, 2, 1)  # [B, N, 3]
    gt = torch.randn(batch_size, 8192, 3)
    partial_coords = partial[..., :3]
    
    losses = loss_fn(pred, gt, partial_coords)
    print(f"Loss components: {losses}")