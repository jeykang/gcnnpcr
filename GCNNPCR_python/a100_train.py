#!/usr/bin/env python
"""
Modified training script for the redesigned anti-collapse model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import math

# Import the redesigned model
from a100_model import (
    create_anticollapsemodel,
    ImprovedLoss,
    AntiCollapsePointCompletion
)

# Import dataset
from minimal_main_4 import S3DISDataset

class AntiCollapseTrainer:
    """Trainer with specific strategies to prevent collapse"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_anticollapsemodel().to(self.device)
        
        # Loss function with strong anti-collapse terms
        self.criterion = ImprovedLoss()
        
        # Optimizer with specific settings for anti-collapse
        self.setup_optimizer()
        
        # Data loaders
        self.setup_data_loaders()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.collapse_history = []
    
    def setup_optimizer(self):
        """Optimizer configuration to prevent collapse"""
        
        # Different learning rates for different components
        params = [
            # Encoder - moderate learning rate
            {'params': self.model.encoder.parameters(), 'lr': 1e-4},
            
            # Decoder - lower learning rate to prevent collapse
            {'params': self.model.decoder.parameters(), 'lr': 5e-5},
            
            # Output normalization - very low learning rate
            {'params': self.model.output_norm.parameters(), 'lr': 1e-5},
        ]
        
        # Use Adam with specific betas for stability
        self.optimizer = optim.Adam(params, betas=(0.9, 0.999), eps=1e-8)
        
        # Learning rate scheduler - ReduceLROnPlateau is good for preventing collapse
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=True
        )
    
    def setup_data_loaders(self):
        """Setup data loaders"""
        train_dataset = S3DISDataset(
            root=self.args.dataset_root,
            mask_ratio=0.3,  # Start with less aggressive masking
            num_points=self.args.num_points,
            split="train",
            patches_per_room=4,
            train_areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
            test_areas=["Area_6"]
        )
        
        val_dataset = S3DISDataset(
            root=self.args.dataset_root,
            mask_ratio=0.3,
            num_points=self.args.num_points,
            split="val",
            patches_per_room=2,
            train_areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
            test_areas=["Area_6"]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
    
    def check_collapse(self, output):
        if output.shape[1] == 3:
            output = output.transpose(1, 2)

        std = output.std(dim=1).mean().item()

        # sample-based minimum distance
        sample_size = min(100, output.shape[1])
        sample_idx = torch.randperm(output.shape[1], device=output.device)[:sample_size]
        sample = output[:, sample_idx]
        dist = torch.cdist(sample, sample)
        dist = dist + torch.eye(sample_size, device=dist.device) * 1e10
        min_dist = dist.min(dim=-1)[0].mean().item()

        # treat NaN or too‑small statistics as collapsed
        if math.isnan(std) or math.isnan(min_dist):
            is_collapsed = True
        else:
            is_collapsed = std < self.args.min_std_threshold or min_dist < 0.01

        return {'is_collapsed': is_collapsed,
                'std': std,
                'min_dist': min_dist}

    
    def train_epoch(self, epoch):
        """Train for one epoch with collapse prevention"""
        self.model.train()
        total_loss = 0
        collapse_count = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            partial = batch['partial'].to(self.device)
            full = batch['full'].to(self.device)
            
            # Forward pass
            output = self.model(partial)
            
            # Check for collapse
            collapse_info = self.check_collapse(output)
            if collapse_info['is_collapsed']:
                collapse_count += 1
                
                # Take immediate action if collapse detected
                if collapse_count > 5:
                    print(f"\n⚠️ Multiple collapses detected! Taking action...")
                    
                    # 1. Reduce learning rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    
                    # 2. Add noise to break symmetry
                    with torch.no_grad():
                        if hasattr(self.model.decoder, 'split_scale'):
                            self.model.decoder.split_scale.data += torch.randn_like(
                                self.model.decoder.split_scale.data
                            ) * 0.01
            
            # Compute loss
            gt_coords = full[..., :3]
            partial_coords = partial[..., :3]
            losses = self.criterion(output, gt_coords, partial_coords)
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping is crucial
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update progress bar
            total_loss += losses['total'].item()
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'chamfer': losses['chamfer'].item(),
                'spread': losses['spread'].item(),
                'std': collapse_info['std'],
                'collapsed': collapse_count
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.collapse_history.append(collapse_count)
        
        return avg_loss, collapse_count
    
    def validate(self):
        """Validation with collapse detection"""
        self.model.eval()
        total_loss = 0
        total_std = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                partial = batch['partial'].to(self.device)
                full = batch['full'].to(self.device)
                
                output = self.model(partial)
                
                # Check collapse
                collapse_info = self.check_collapse(output)
                total_std += collapse_info['std']
                
                # Compute loss
                gt_coords = full[..., :3]
                partial_coords = partial[..., :3]
                losses = self.criterion(output, gt_coords, partial_coords)
                
                total_loss += losses['total'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_std = total_std / len(self.val_loader)
        
        return avg_loss, avg_std
    
    def train(self):
        """Main training loop"""
        print("Starting training with anti-collapse architecture...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Encoder: {self.model.encoder_type}")
        print(f"Decoder: {self.model.decoder_type}")
        
        for epoch in range(self.args.num_epochs):
            # Train
            train_loss, collapse_count = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_std = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Collapses: {collapse_count}")
            print(f"  Val Loss: {val_loss:.4f}, Val Std: {val_std:.4f}")
            
            # Check if we need intervention
            if val_std < 0.15:
                print("⚠️ WARNING: Low validation spread detected!")
                print("  Applying intervention...")
                
                # Reinitialize parts of decoder if using folding
                if hasattr(self.model.decoder, 'folding_nets'):
                    for net in self.model.decoder.folding_nets:
                        for layer in net:
                            if isinstance(layer, nn.Linear):
                                nn.init.xavier_normal_(layer.weight)
                                if layer.bias is not None:
                                    nn.init.zeros_(layer.bias)
                
                # Increase learning rate temporarily
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 2, 1e-3)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_std': val_std
                }, 'best_model_anticollapse.pth')
                print(f"  ✓ Saved best model (loss: {val_loss:.4f}, std: {val_std:.4f})")


# Training configuration specifically for anti-collapse
class AntiCollapseConfig:
    def __init__(self):
        self.dataset_root = "/data/S3DIS"
        self.num_points = 8192
        self.batch_size = 8  # Smaller batch size for stability
        self.num_epochs = 200
        self.learning_rate = 1e-4
        
        # Progressive training schedule
        self.progressive_schedule = {
            0: {'mask_ratio': 0.2, 'lr_mult': 1.0},
            20: {'mask_ratio': 0.3, 'lr_mult': 0.8},
            40: {'mask_ratio': 0.4, 'lr_mult': 0.6},
            60: {'mask_ratio': 0.5, 'lr_mult': 0.4},
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AntiCollapseTrainer(args)
    
    # Train
    trainer.train()