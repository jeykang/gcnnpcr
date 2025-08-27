#!/usr/bin/env python
"""
Training script for the enhanced model architecture
Can also continue training the old model with better hyperparameters
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np

# Import dataset from original
from minimal_main_4 import S3DISDataset, save_point_cloud_comparison

# Import enhanced model
from enhanced_model import (
    EnhancedPointCompletionModel, 
    CombinedLossWithEMD,
    create_enhanced_model
)

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_losses = {'total': 0, 'chamfer': 0, 'repulsion': 0, 'smoothness': 0, 'coverage': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        partial = batch['partial'].to(device)
        full = batch['full'].to(device)
        
        # Forward pass
        pred = model(partial)  # [B, 3, N]
        pred = pred.permute(0, 2, 1)  # [B, N, 3]
        
        # Compute loss
        gt_coords = full[..., :3]
        partial_coords = partial[..., :3]
        
        # Filter out masked points from partial
        B = partial_coords.shape[0]
        partial_filtered = []
        for b in range(B):
            mask = (partial_coords[b].abs().sum(dim=-1) > 1e-6)
            partial_filtered.append(partial_coords[b][mask])
        
        # Compute losses
        losses = loss_fn(pred, gt_coords, partial_coords)
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'cd': losses['chamfer'].item()
        })
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_losses = {'total': 0, 'chamfer': 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            partial = batch['partial'].to(device)
            full = batch['full'].to(device)
            
            pred = model(partial)
            pred = pred.permute(0, 2, 1)
            
            gt_coords = full[..., :3]
            partial_coords = partial[..., :3]
            
            losses = loss_fn(pred, gt_coords, partial_coords)
            
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            num_batches += 1
    
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses

def main():
    parser = argparse.ArgumentParser(description="Train enhanced point completion model")
    
    # Model selection
    parser.add_argument("--model", choices=['enhanced', 'original'], default='enhanced',
                       help="Which model architecture to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Dataset
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=8192)
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=['cosine', 'step', 'plateau'], default='plateau')
    
    # Loss weights (for enhanced model)
    parser.add_argument("--chamfer_weight", type=float, default=1.0)
    parser.add_argument("--repulsion_weight", type=float, default=0.05)
    parser.add_argument("--smoothness_weight", type=float, default=0.01)
    parser.add_argument("--coverage_weight", type=float, default=0.2)
    
    # Output
    parser.add_argument("--save_dir", type=str, default="enhanced_training")
    parser.add_argument("--vis_interval", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.save_dir}/visuals", exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model == 'enhanced':
        print("Using enhanced architecture")
        model = create_enhanced_model(args.checkpoint)
        loss_fn = CombinedLossWithEMD(
            chamfer_weight=args.chamfer_weight,
            repulsion_weight=args.repulsion_weight,
            smoothness_weight=args.smoothness_weight,
            coverage_weight=args.coverage_weight
        )
    else:
        print("Using original architecture")
        from minimal_main_4 import FullModelSnowflake
        model = FullModelSnowflake(
            g_hidden_dims=[64, 128],
            g_out_dim=128,
            t_d_model=128,
            t_nhead=8,
            t_layers=4,
            coarse_num=64,
            use_attention_encoder=True,
            radius=1.0,
        )
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        # Use enhanced loss even for original model
        loss_fn = CombinedLossWithEMD(
            chamfer_weight=1.0,
            repulsion_weight=0.1,  # Higher for original model
            smoothness_weight=0.01,
            coverage_weight=0.1
        )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = S3DISDataset(
        root=args.dataset_root,
        mask_ratio=0.4,  # Less aggressive masking
        num_points=args.num_points,
        split="train",
        normal_k=16,
        patches_per_room=4,
        train_areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        test_areas=["Area_6"]
    )
    
    val_dataset = S3DISDataset(
        root=args.dataset_root,
        mask_ratio=0.4,
        num_points=args.num_points,
        split="val",
        normal_k=16,
        patches_per_room=2,
        train_areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        test_areas=["Area_6"]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, 
                           weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:  # plateau
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"CD: {train_losses['chamfer']:.4f}, "
              f"Rep: {train_losses['repulsion']:.4f}")
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"CD: {val_losses['chamfer']:.4f}")
        
        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_losses['total'])
        else:
            scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        
        torch.save(checkpoint, f"{args.save_dir}/checkpoints/latest.pth")
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(checkpoint, f"{args.save_dir}/checkpoints/best.pth")
            print(f"★ New best model (val loss: {best_val_loss:.4f})")
        
        # Generate visualization
        if epoch % args.vis_interval == 0:
            print("Generating visualization...")
            model.eval()
            with torch.no_grad():
                sample = next(iter(val_loader))
                partial = sample['partial'][:1].to(device)
                full = sample['full'][:1].to(device)
                
                pred = model(partial)
                pred = pred.permute(0, 2, 1)[0].cpu()
                
                partial_coords = partial[0, ..., :3].cpu()
                mask = (partial_coords.abs().sum(dim=-1) > 1e-6)
                partial_filtered = partial_coords[mask]
                
                original = full[0, ..., :3].cpu()
                
                save_point_cloud_comparison(
                    partial_filtered, pred, original, 
                    epoch, f"{args.save_dir}/visuals"
                )
                
                # Also save raw arrays
                os.makedirs(f"{args.save_dir}/visuals/epoch_{epoch}", exist_ok=True)
                np.save(f"{args.save_dir}/visuals/epoch_{epoch}/partial.npy", partial_filtered.numpy())
                np.save(f"{args.save_dir}/visuals/epoch_{epoch}/completed.npy", pred.numpy())
                np.save(f"{args.save_dir}/visuals/epoch_{epoch}/original.npy", original.numpy())
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()