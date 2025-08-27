#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed training script for GCNN Point Cloud Reconstruction
Supports both single-node multi-GPU and multi-node training via PyTorch DDP
"""

import os
import sys
import random
import time
import argparse
import json
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Import the model and dataset from minimal_main_4
from minimal_main_4 import (
    FullModelSnowflake,
    S3DISDataset,
    combined_loss,
    chamfer_distance,
    save_point_cloud_comparison
)

def setup_distributed():
    """Initialize distributed training environment"""
    
    # Check if we're in a distributed environment
    # torchrun sets these environment variables
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Detected torchrun environment: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment (when not using torchrun)
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set up master address and port
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 
                                                    os.environ.get('SLURM_SUBMIT_HOST', '127.0.0.1'))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        # Set environment variables for torch.distributed
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        print(f"Detected SLURM environment: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    else:
        # Single GPU mode
        world_size = 1
        rank = 0
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        print(f"Single GPU mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # Initialize process group
    if world_size > 1:
        # Initialize the process group
        # torchrun will have already set up the rendezvous backend
        if 'TORCHELASTIC_RESTART_COUNT' in os.environ:
            # torchrun/torch.distributed.launch is being used
            dist.init_process_group(backend='nccl')
        else:
            # Manual initialization
            dist.init_process_group(backend='nccl', 
                                  init_method='env://',
                                  world_size=world_size, 
                                  rank=rank)
    
    # Handle CUDA device assignment properly
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # Use local_rank to select device
            # When CUDA_VISIBLE_DEVICES is set, devices are already filtered
            device_id = local_rank % num_gpus
            torch.cuda.set_device(device_id)
            
            # Get actual device name for logging
            device_name = torch.cuda.get_device_name(device_id)
            print(f"[Rank {rank}] Using CUDA device {device_id}: {device_name}")
            print(f"[Rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        else:
            print(f"[Rank {rank}] WARNING: No CUDA devices available!")
            raise RuntimeError("No CUDA devices available")
    else:
        print(f"[Rank {rank}] WARNING: CUDA not available, using CPU")
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    """Check if this is the main process (rank 0)"""
    return rank == 0

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path, rank):
    """Save model checkpoint (only from rank 0)"""
    if not is_main_process(rank):
        return
    
    # Handle DDP model state dict
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"[Rank {rank}] Saved checkpoint to {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle DDP model
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if is_main_process(rank):
        print(f"[Rank {rank}] Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return checkpoint['epoch']

def train_epoch(model, train_loader, optimizer, device, rank, world_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar only on main process
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Training [Rank {rank}]")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        partial_6d = batch["partial"].to(device)
        full_6d = batch["full"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        completed = model(partial_6d)  # [B,3,8192]
        completed_coords = completed.permute(0, 2, 1).contiguous()  # [B,8192,3]
        gt_coords = full_6d[..., :3]
        
        # Compute loss
        loss = combined_loss(completed_coords, gt_coords)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        if is_main_process(rank) and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': loss.item()})
    
    # Average loss across all processes
    avg_loss = total_loss / num_batches
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss).cuda()
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()
    
    return avg_loss

def validate(model, val_loader, device, rank, world_size):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            partial_val_6d = batch["partial"].to(device)
            full_val_6d = batch["full"].to(device)
            
            completed_val = model(partial_val_6d)
            completed_val_coords = completed_val.permute(0, 2, 1).contiguous()
            gt_val_coords = full_val_6d[..., :3]
            
            loss = chamfer_distance(completed_val_coords, gt_val_coords)
            total_loss += loss.item()
            num_batches += 1
    
    # Average loss across all processes
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss).cuda()
        avg_loss_tensor = reduce_tensor(avg_loss_tensor, world_size)
        avg_loss = avg_loss_tensor.item()
    
    return avg_loss

def main(args):
    """Main training function"""
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        print(f"[Rank {rank}] WARNING: Running on CPU!")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed + rank)  # Different seed per rank
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    
    # Print configuration (only from main process)
    if is_main_process(rank):
        print(f"Starting distributed training with {world_size} GPUs")
        print(f"Dataset root: {args.dataset_root}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
    
    # Create datasets
    train_areas = args.train_areas.split(',') if args.train_areas else ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"]
    test_areas = args.test_areas.split(',') if args.test_areas else ["Area_6"]
    
    train_dataset = S3DISDataset(
        root=args.dataset_root,
        mask_ratio=args.mask_ratio,
        num_points=args.num_points,
        split="train",
        normal_k=16,
        patches_per_room=args.patches_per_room,
        train_areas=train_areas,
        test_areas=test_areas,
    )
    
    val_dataset = S3DISDataset(
        root=args.dataset_root,
        mask_ratio=args.mask_ratio,
        num_points=args.num_points,
        split="val",
        normal_k=16,
        patches_per_room=args.patches_per_room,
        train_areas=train_areas,
        test_areas=test_areas,
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = FullModelSnowflake(
        g_hidden_dims=[64, 128],
        g_out_dim=128,
        t_d_model=128,
        t_nhead=8,
        t_layers=4,
        coarse_num=64,
        use_attention_encoder=args.use_attention_encoder,
        radius=1.0,
    ).to(device)
    
    # Wrap model with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Print model info (only from main process)
    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume and args.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path, rank)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.num_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        if is_main_process(rank):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, rank, world_size)
        
        # Validate
        val_loss = validate(model, val_loader, device, rank, world_size)
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch summary (only from main process)
        if is_main_process(rank):
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = args.checkpoint_dir or "checkpoints"
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch+1, checkpoint_path, rank)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, epoch+1, best_checkpoint_path, rank)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Generate visualization
            if (epoch + 1) % args.vis_interval == 0:
                # Get a sample for visualization
                sample_batch = next(iter(val_loader))
                partial_6d = sample_batch["partial"].to(device)
                full_6d = sample_batch["full"].to(device)
                
                model.eval()
                with torch.no_grad():
                    completed = model(partial_6d[:1])  # Use first sample
                    completed_coords = completed.permute(0, 2, 1).contiguous()
                
                partial_0 = partial_6d[0, ..., :3].detach().cpu()
                completed_0 = completed_coords[0].detach().cpu()
                original_0 = full_6d[0, ..., :3].detach().cpu()
                save_point_cloud_comparison(partial_0, completed_0, original_0, epoch+1, args.vis_dir or "visuals")
        
        # Synchronize all processes
        if world_size > 1:
            dist.barrier()
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process(rank):
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training for GCNN Point Cloud Reconstruction")
    
    # Dataset arguments
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to S3DIS dataset")
    parser.add_argument("--train_areas", type=str, default="", help="Comma-separated list of training areas")
    parser.add_argument("--test_areas", type=str, default="", help="Comma-separated list of test areas")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for point clouds")
    parser.add_argument("--num_points", type=int, default=8192, help="Number of points per sample")
    parser.add_argument("--patches_per_room", type=int, default=4, help="Number of patches to extract per room")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--use_attention_encoder", action="store_true", help="Use attention in encoder")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume from")
    
    # Visualization arguments
    parser.add_argument("--vis_interval", type=int, default=10, help="Visualization interval (epochs)")
    parser.add_argument("--vis_dir", type=str, default="visuals", help="Directory to save visualizations")
    
    args = parser.parse_args()
    main(args)