#!/usr/bin/env python
"""
Optimized training script for 8x NVIDIA A100 GPUs
Includes mixed precision, gradient accumulation, and progressive training
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

# Import dataset and visualization
from minimal_main_4 import S3DISDataset, save_point_cloud_comparison

# Import the new model
from a100_model import (
    create_multigpu_model,
    MultiGPULoss
)


class ProgressiveDataset(S3DISDataset):
    """Dataset wrapper for progressive training (start with fewer points)"""
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
        
        # Set up experiment tracking (optional)
        if self.is_main_process and args.use_wandb:
            wandb.init(project="point-completion-multigpu", config=args)
        
        # Create model
        self.model = create_multigpu_model(args.checkpoint).to(self.device)
        
        # Wrap with DDP
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], 
                           find_unused_parameters=False)  # More efficient
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=args.use_amp)
        
        # Loss function
        self.criterion = MultiGPULoss(
            chamfer_weight=args.chamfer_weight,
            emd_weight=args.emd_weight,
            repulsion_weight=args.repulsion_weight,
            coverage_weight=args.coverage_weight,
            smoothness_weight=args.smoothness_weight,
            use_emd=args.use_emd
        )

        # Training state (must be before data loader setup)
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.current_points = args.start_points if args.progressive else args.num_points
        
        # Optimizer with different learning rates for different parts
        self.setup_optimizer()
        
        # Data loaders (must be before scheduler for OneCycle)
        self.setup_data_loaders()
        
        # Learning rate scheduler (needs train_loader for OneCycle)
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
        params.append({'params': model.encoder.parameters(), 
                      'lr': self.args.learning_rate * 0.1})
        params.append({'params': model.transformer.parameters(), 
                      'lr': self.args.learning_rate})
        params.append({'params': model.decoder.parameters(), 
                      'lr': self.args.learning_rate * 2})
        
        # Use AdamW with better weight decay
        self.optimizer = optim.AdamW(params, 
                                    lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay,
                                    betas=(0.9, 0.999))
    
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
                pct_start=0.1,  # 10% warmup
                anneal_strategy='cos'
            )
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
    
    def setup_data_loaders(self):
        """Setup data loaders with proper batch size for A100s"""
        # Default area splits
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
        
        # Loaders with optimized settings
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=2  # Prefetch batches
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
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
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
                output = output.permute(0, 2, 1)  # [B, N, 3]
                
                gt_coords = full[..., :3]
                partial_coords = partial[..., :3]
                
                # Compute loss
                losses = self.criterion(output, gt_coords, partial_coords)
                loss = losses['total'] / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              max_norm=self.args.grad_clip)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update scheduler (if using OneCycle)
                if self.args.scheduler == 'onecycle':
                    self.scheduler.step()
            
            total_loss += losses['total'].item()
            num_batches += 1
            
            # Update progress bar
            if self.is_main_process and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': losses['total'].item(),
                    'cd': losses.get('chamfer', 0).item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Average loss
        avg_loss = total_loss / num_batches
        
        # Reduce across GPUs
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                partial = batch['partial'].to(self.device, non_blocking=True)
                full = batch['full'].to(self.device, non_blocking=True)
                
                with autocast(enabled=self.args.use_amp):
                    output = self.model(partial)
                    output = output.permute(0, 2, 1)
                    
                    gt_coords = full[..., :3]
                    partial_coords = partial[..., :3]
                    
                    losses = self.criterion(output, gt_coords, partial_coords)
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Reduce across GPUs
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor)
            avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
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
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch + 1
            
            # Progressive training: increase points over time
            if self.args.progressive:
                progress = epoch / self.args.num_epochs
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
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Update scheduler (if not OneCycle)
            if self.args.scheduler != 'onecycle':
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            if self.is_main_process:
                print(f"Epoch {self.epoch}/{self.args.num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Log to wandb
                if self.args.use_wandb:
                    wandb.log({
                        'epoch': self.epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'current_points': self.current_points
                    })
                
                # Save checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint(is_best)
                
                # Visualization
                if epoch % self.args.vis_interval == 0:
                    self.visualize()
            
            # Synchronize
            if self.world_size > 1:
                dist.barrier()
    
    def visualize(self):
        """Generate visualization"""
        if not self.is_main_process:
            return
        
        self.model.eval()
        sample = next(iter(self.val_loader))
        
        with torch.no_grad():
            partial = sample['partial'][:1].to(self.device)
            full = sample['full'][:1].to(self.device)
            
            with autocast(enabled=self.args.use_amp):
                output = self.model(partial)
            
            output = output.permute(0, 2, 1)[0].cpu()
            partial_viz = partial[0, ..., :3].cpu()
            original = full[0, ..., :3].cpu()
            
            # Filter out masked points
            mask = partial_viz.abs().sum(dim=-1) > 1e-6
            partial_viz = partial_viz[mask]
            
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
    parser.add_argument('--batch_size', type=int, default=16)  # Per GPU - use 16 for A100
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Progressive training
    parser.add_argument('--progressive', action='store_true')
    parser.add_argument('--start_points', type=int, default=1024)
    
    # Loss weights
    parser.add_argument('--chamfer_weight', type=float, default=1.0)
    parser.add_argument('--emd_weight', type=float, default=0.5)
    parser.add_argument('--repulsion_weight', type=float, default=0.1)
    parser.add_argument('--coverage_weight', type=float, default=0.2)
    parser.add_argument('--smoothness_weight', type=float, default=0.05)
    parser.add_argument('--use_emd', action='store_true')
    
    # Optimizer
    parser.add_argument('--scheduler', choices=['cosine', 'onecycle', 'plateau'], 
                       default='onecycle')
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_multigpu')
    parser.add_argument('--vis_dir', type=str, default='visuals_multigpu')
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Create trainer
    trainer = Trainer(args, rank, local_rank, world_size)
    
    # Train
    trainer.train()
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()