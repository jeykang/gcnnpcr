#!/usr/bin/env python
"""
Optimized training script for 8x NVIDIA A100 GPUs
Fixed version with anti-collapse monitoring and enhanced loss
Compatible with PyTorch 1.13.0
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np
import argparse
import json
from typing import Dict, Any

# Import dataset and visualization
from minimal_main_4 import S3DISDataset, save_point_cloud_comparison

# Import the fixed model
from a100_model import (
    create_multigpu_model,
    EnhancedMultiGPULoss
)


class CollapseMonitor:
    """Monitor for detecting and handling point cloud collapse"""
    def __init__(self, min_std_threshold: float = 0.1, 
                 patience: int = 5,
                 recovery_enabled: bool = True):
        self.min_std_threshold = min_std_threshold
        self.patience = patience
        self.recovery_enabled = recovery_enabled
        self.collapse_counter = 0
        self.history = []
    
    def check_collapse(self, output_points: torch.Tensor) -> Dict[str, Any]:
        """Check if points have collapsed"""
        B, N, _ = output_points.shape
        
        # Compute per-batch statistics
        std_per_batch = output_points.std(dim=1).mean(dim=-1)  # [B]
        mean_std = std_per_batch.mean().item()
        min_std = std_per_batch.min().item()
        
        # Check pairwise distances for more robust detection
        sample_size = min(1000, N)
        sample_idx = torch.randperm(N, device=output_points.device)[:sample_size]
        sample_points = output_points[:, sample_idx]
        
        pairwise_dist = torch.cdist(sample_points, sample_points)
        # Remove diagonal
        mask = ~torch.eye(sample_size, dtype=torch.bool, device=pairwise_dist.device)
        pairwise_dist = pairwise_dist[mask].reshape(B, sample_size, sample_size - 1)
        
        mean_nn_dist = pairwise_dist.min(dim=-1)[0].mean().item()
        
        # Determine collapse state
        is_collapsed = mean_std < self.min_std_threshold or mean_nn_dist < 0.01
        
        if is_collapsed:
            self.collapse_counter += 1
        else:
            self.collapse_counter = 0
        
        # Store history
        self.history.append({
            'mean_std': mean_std,
            'min_std': min_std,
            'mean_nn_dist': mean_nn_dist,
            'is_collapsed': is_collapsed
        })
        
        return {
            'is_collapsed': is_collapsed,
            'mean_std': mean_std,
            'min_std': min_std,
            'mean_nn_dist': mean_nn_dist,
            'consecutive_collapses': self.collapse_counter,
            'needs_intervention': self.collapse_counter >= self.patience
        }
    
    def get_recovery_action(self, model, optimizer):
        """Get recovery action if collapse is detected"""
        if not self.recovery_enabled:
            return None
        
        actions = []
        
        # 1. Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        actions.append("Reduced learning rate by 50%")
        
        # 2. Reinitialize decoder seed points if using old architecture
        if hasattr(model, 'module'):
            decoder = model.module.decoder if hasattr(model.module, 'decoder') else None
        else:
            decoder = model.decoder if hasattr(model, 'decoder') else None
        
        if decoder and hasattr(decoder, 'seed_generator'):
            # For new architecture, reinitialize folding network
            for param in decoder.seed_generator.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
            actions.append("Reinitialized decoder seed generator")
        
        # 3. Reset collapse counter
        self.collapse_counter = 0
        
        return actions


class ProgressiveDataset(S3DISDataset):
    """Dataset wrapper for progressive training"""
    def __init__(self, *args, current_points=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_points = current_points if current_points is not None else self.num_points
    
    def set_num_points(self, num_points):
        """Dynamically adjust number of points during training"""
        self.current_points = min(num_points, self.num_points)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Subsample if in progressive mode
        if self.current_points < self.num_points:
            N = data['partial'].shape[0]
            indices = torch.randperm(N)[:self.current_points]
            data['partial'] = data['partial'][indices]
            data['full'] = data['full'][indices]
        
        return data


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, args, rank, local_rank, world_size):
        self.args = args
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        
        # Collapse monitoring
        self.collapse_monitor = CollapseMonitor(
            min_std_threshold=0.15,  # Threshold for detecting collapse
            patience=5,  # How many epochs to wait before intervention
            recovery_enabled=True
        )
        
        # Create model
        self.model = create_multigpu_model(args.checkpoint).to(self.device)
        
        # Wrap with DDP
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], 
                           find_unused_parameters=False)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=args.use_amp)
        
        # Enhanced loss function with diversity term
        self.criterion = EnhancedMultiGPULoss(
            chamfer_weight=args.chamfer_weight,
            emd_weight=args.emd_weight,
            repulsion_weight=args.repulsion_weight,
            coverage_weight=args.coverage_weight,
            smoothness_weight=args.smoothness_weight,
            diversity_weight=args.diversity_weight,  # New parameter
            min_spread=0.3,  # Minimum acceptable spread
            use_emd=args.use_emd
        )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.current_points = args.start_points if args.progressive else args.num_points
        
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'collapse_events': [],
            'point_spread': [],
            'learning_rate': []
        }
        
        # Optimizer with different learning rates for different parts
        self.setup_optimizer()
        
        # Data loaders
        self.setup_data_loaders()
        
        # Learning rate scheduler
        self.setup_scheduler()
    
    @property
    def is_main_process(self):
        return self.rank == 0
    
    def setup_optimizer(self):
        """Setup optimizer with layer-wise learning rates"""
        params = []
        
        # Get model (handle DDP wrapper)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Different LR for encoder, transformer, decoder
        # Lower LR for decoder to prevent collapse
        params.append({'params': model.encoder.parameters(), 
                      'lr': self.args.learning_rate * 0.5})
        params.append({'params': model.transformer.parameters(), 
                      'lr': self.args.learning_rate})
        params.append({'params': model.decoder.parameters(), 
                      'lr': self.args.learning_rate * 0.5})  # Reduced from 2x
        
        # Use AdamW with better weight decay
        self.optimizer = optim.AdamW(params, 
                                    lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay,
                                    betas=(0.9, 0.999),
                                    eps=1e-8)
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
            )
        elif self.args.scheduler == 'onecycle':
            steps_per_epoch = len(self.train_loader)
            self.scheduler = lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=steps_per_epoch * self.args.num_epochs,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=25,  # Less aggressive warmup
                final_div_factor=10000
            )
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10,
                min_lr=1e-7, verbose=True
            )
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        train_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"]
        test_areas = ["Area_6"]
        
        # Datasets
        train_dataset = ProgressiveDataset(
            root=self.args.dataset_root,
            mask_ratio=self.args.mask_ratio,
            num_points=self.args.num_points,
            split="train",
            patches_per_room=self.args.patches_per_room,
            current_points=self.current_points,
            train_areas=train_areas,
            test_areas=test_areas
        )
        
        val_dataset = S3DISDataset(
            root=self.args.dataset_root,
            mask_ratio=self.args.mask_ratio,
            num_points=self.args.num_points,
            split="val",
            patches_per_room=2,
            train_areas=train_areas,
            test_areas=test_areas
        )
        
        # Samplers
        train_sampler = DistributedSampler(train_dataset, 
                                          num_replicas=self.world_size,
                                          rank=self.rank,
                                          shuffle=True) if self.world_size > 1 else None
        
        val_sampler = DistributedSampler(val_dataset,
                                        num_replicas=self.world_size,
                                        rank=self.rank,
                                        shuffle=False) if self.world_size > 1 else None
        
        # Loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def train_epoch(self):
        """Train for one epoch with collapse monitoring"""
        self.model.train()
        total_losses = {key: 0.0 for key in ['total', 'chamfer', 'repulsion', 'diversity', 'coverage']}
        num_batches = 0
        collapse_events = 0
        
        # Progress bar
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = self.train_loader
        
        # Gradient accumulation
        accumulation_steps = self.args.gradient_accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            partial = batch['partial'].to(self.device, non_blocking=True)
            full = batch['full'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast(enabled=self.args.use_amp):
                output = self.model(partial)  # [B, 3, N]
                
                # Check for NaN/Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"Warning: NaN/Inf detected in output at batch {batch_idx}")
                    continue
                
                # Prepare for loss computation
                output_points = output.transpose(1, 2) if output.shape[1] == 3 else output
                gt_coords = full[..., :3]
                partial_coords = partial[..., :3]
                
                # Check for collapse
                collapse_info = self.collapse_monitor.check_collapse(output_points)
                if collapse_info['is_collapsed']:
                    collapse_events += 1
                    if self.is_main_process:
                        print(f"\n⚠️ Collapse detected! Std: {collapse_info['mean_std']:.4f}, "
                              f"NN dist: {collapse_info['mean_nn_dist']:.4f}")
                
                # Compute loss
                losses = self.criterion(output_points, gt_coords, partial_coords)
                loss = losses['total'] / accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.args.grad_clip
                )
                
                # Check for gradient explosion
                if grad_norm > self.args.grad_clip * 10:
                    print(f"Warning: Large gradient norm {grad_norm:.2f}")
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update scheduler (if using OneCycle)
                if self.args.scheduler == 'onecycle':
                    self.scheduler.step()
            
            # Accumulate losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': losses['total'].item(),
                    'cd': losses.get('chamfer', 0).item(),
                    'div': losses.get('diversity', 0).item(),
                    'std': collapse_info['mean_std'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Handle collapse if persistent
        if self.collapse_monitor.collapse_counter >= self.collapse_monitor.patience:
            if self.is_main_process:
                print("\n🚨 Persistent collapse detected! Taking recovery action...")
                actions = self.collapse_monitor.get_recovery_action(self.model, self.optimizer)
                for action in actions:
                    print(f"  - {action}")
        
        # Average losses
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        
        # Reduce across GPUs
        if self.world_size > 1:
            for key in avg_losses:
                loss_tensor = torch.tensor(avg_losses[key]).to(self.device)
                dist.all_reduce(loss_tensor)
                avg_losses[key] = loss_tensor.item() / self.world_size
        
        # Record metrics
        self.metrics_history['collapse_events'].append(collapse_events)
        
        return avg_losses, collapse_events
    
    def validate(self):
        """Validation loop with collapse detection"""
        self.model.eval()
        total_losses = {key: 0.0 for key in ['total', 'chamfer', 'diversity']}
        num_batches = 0
        mean_spreads = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                partial = batch['partial'].to(self.device, non_blocking=True)
                full = batch['full'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.args.use_amp):
                    output = self.model(partial)
                    
                    # Check for NaN/Inf
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        continue
                    
                    output_points = output.transpose(1, 2) if output.shape[1] == 3 else output
                    gt_coords = full[..., :3]
                    partial_coords = partial[..., :3]
                    
                    # Track spread
                    spread = output_points.std(dim=1).mean().item()
                    mean_spreads.append(spread)
                    
                    losses = self.criterion(output_points, gt_coords, partial_coords)
                
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
                num_batches += 1
        
        avg_losses = {key: val / num_batches for key, val in total_losses.items()}
        avg_spread = np.mean(mean_spreads) if mean_spreads else 0.0
        
        # Reduce across GPUs
        if self.world_size > 1:
            for key in avg_losses:
                loss_tensor = torch.tensor(avg_losses[key]).to(self.device)
                dist.all_reduce(loss_tensor)
                avg_losses[key] = loss_tensor.item() / self.world_size
            
            spread_tensor = torch.tensor(avg_spread).to(self.device)
            dist.all_reduce(spread_tensor)
            avg_spread = spread_tensor.item() / self.world_size
        
        self.metrics_history['point_spread'].append(avg_spread)
        
        return avg_losses, avg_spread
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint with metrics"""
        if not self.is_main_process:
            return
        
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') \
                     else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history,
            'collapse_monitor_history': self.collapse_monitor.history,
            'args': self.args
        }
        
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        # Save regular checkpoint
        path = os.path.join(self.args.checkpoint_dir, f'epoch_{self.epoch}.pth')
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {self.best_val_loss:.4f}")
        
        # Save metrics history as JSON for analysis
        metrics_path = os.path.join(self.args.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def train(self):
        """Main training loop with enhanced monitoring"""
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch + 1
            
            # Progressive training
            if self.args.progressive:
                progress = epoch / self.args.num_epochs
                # Use smoother progression
                progress = np.sin(progress * np.pi / 2)  # Sine curve for smoother increase
                target_points = int(self.args.start_points + 
                                  (self.args.num_points - self.args.start_points) * progress)
                target_points = min(target_points, self.args.num_points)
                
                if target_points != self.current_points:
                    self.current_points = target_points
                    self.train_dataset.set_num_points(target_points)
                    if self.is_main_process:
                        print(f"Progressive training: using {target_points} points")
            
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training
            train_losses, collapse_events = self.train_epoch()
            
            # Validation
            val_losses, avg_spread = self.validate()
            
            # Update scheduler (if not OneCycle)
            if self.args.scheduler != 'onecycle':
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.metrics_history['train_loss'].append(train_losses['total'])
            self.metrics_history['val_loss'].append(val_losses['total'])
            self.metrics_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            if self.is_main_process:
                print(f"\n{'='*60}")
                print(f"Epoch {self.epoch}/{self.args.num_epochs}")
                print(f"Train - Total: {train_losses['total']:.4f}, "
                      f"CD: {train_losses['chamfer']:.4f}, "
                      f"Div: {train_losses['diversity']:.4f}")
                print(f"Val   - Total: {val_losses['total']:.4f}, "
                      f"CD: {val_losses['chamfer']:.4f}, "
                      f"Spread: {avg_spread:.4f}")
                print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                      f"Collapses: {collapse_events}")
                
                # Warning if spread is low
                if avg_spread < 0.2:
                    print("⚠️ WARNING: Low point cloud spread detected!")
                
                # Save checkpoint
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                self.save_checkpoint(is_best)
                
                # Visualization
                if epoch % self.args.vis_interval == 0:
                    self.visualize()
            
            # Synchronize
            if self.world_size > 1:
                dist.barrier()
    
    def visualize(self):
        """Generate visualization with diversity check"""
        if not self.is_main_process:
            return
        
        self.model.eval()
        sample = next(iter(self.val_loader))
        
        with torch.no_grad():
            partial = sample['partial'][:1].to(self.device)
            full = sample['full'][:1].to(self.device)
            
            with autocast(enabled=self.args.use_amp):
                output = self.model(partial)
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Warning: NaN/Inf in visualization output")
                return
            
            output = output.permute(0, 2, 1)[0].cpu()
            partial_viz = partial[0, ..., :3].cpu()
            original = full[0, ..., :3].cpu()
            
            # Filter out masked points
            mask = partial_viz.abs().sum(dim=-1) > 1e-6
            partial_viz = partial_viz[mask]
            
            # Check output quality
            output_std = output.std(dim=0).mean().item()
            print(f"Visualization - Output spread: {output_std:.4f}")
            
            os.makedirs(self.args.vis_dir, exist_ok=True)
            save_point_cloud_comparison(
                partial_viz, output, original,
                self.epoch, self.args.vis_dir
            )


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--patches_per_room', type=int, default=8)
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)  # Reduced
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Progressive training
    parser.add_argument('--progressive', action='store_true')
    parser.add_argument('--start_points', type=int, default=2048)  # Start higher
    
    # Loss weights
    parser.add_argument('--chamfer_weight', type=float, default=1.0)
    parser.add_argument('--emd_weight', type=float, default=0.3)
    parser.add_argument('--repulsion_weight', type=float, default=0.2)  # Increased
    parser.add_argument('--coverage_weight', type=float, default=0.2)
    parser.add_argument('--smoothness_weight', type=float, default=0.05)
    parser.add_argument('--diversity_weight', type=float, default=0.3)  # New
    parser.add_argument('--use_emd', action='store_true')
    
    # Optimizer
    parser.add_argument('--scheduler', choices=['cosine', 'onecycle', 'plateau'], 
                       default='plateau')  # Changed default
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_fixed')
    parser.add_argument('--vis_dir', type=str, default='visuals_fixed')
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print("="*60)
        print("FIXED TRAINING WITH ANTI-COLLAPSE ARCHITECTURE")
        print("="*60)
        print(f"Key improvements:")
        print(f"  - Multiple local features instead of single global")
        print(f"  - Folding-based decoder with spatial conditioning")
        print(f"  - Learnable offset scales (not diminishing)")
        print(f"  - Diversity loss weight: {args.diversity_weight}")
        print(f"  - Collapse monitoring enabled")
        print("="*60)
    
    # Create trainer
    trainer = Trainer(args, rank, local_rank, world_size)
    
    # Train
    trainer.train()
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()