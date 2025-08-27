#!/usr/bin/env python
"""
Script to analyze training progress and determine if the model is learning properly
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re

def load_training_logs(log_dir="logs"):
    """Parse training logs to extract metrics"""
    main_log = os.path.join(log_dir, "rank_0_detailed.log")
    
    if not os.path.exists(main_log):
        print(f"No log file found at {main_log}")
        return None
    
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []
    
    with open(main_log, 'r') as f:
        for line in f:
            # Extract epoch
            epoch_match = re.search(r'Epoch (\d+)/\d+', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Extract losses
            loss_match = re.search(r'Train Loss: ([\d.]+), Val Loss: ([\d.]+)', line)
            if loss_match:
                epochs.append(current_epoch)
                train_losses.append(float(loss_match.group(1)))
                val_losses.append(float(loss_match.group(2)))
            
            # Extract learning rate
            lr_match = re.search(r'Learning rate: ([\d.e-]+)', line)
            if lr_match and len(epochs) == len(learning_rates) + 1:
                learning_rates.append(float(lr_match.group(1)))
    
    if not epochs:
        print("No training metrics found in logs")
        return None
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rate': learning_rates[:len(epochs)]
    }

def analyze_visualizations(vis_dir="visuals"):
    """Analyze saved point clouds to check quality"""
    import glob
    
    epoch_dirs = sorted(glob.glob(os.path.join(vis_dir, "epoch_*")))
    
    if not epoch_dirs:
        print(f"No visualization epochs found in {vis_dir}")
        return None
    
    metrics = []
    
    for epoch_dir in epoch_dirs:
        epoch_num = int(os.path.basename(epoch_dir).replace("epoch_", ""))
        
        partial_path = os.path.join(epoch_dir, "partial.npy")
        completed_path = os.path.join(epoch_dir, "completed.npy")
        original_path = os.path.join(epoch_dir, "original.npy")
        
        if not all(os.path.exists(p) for p in [partial_path, completed_path, original_path]):
            continue
        
        partial = np.load(partial_path)
        completed = np.load(completed_path)
        original = np.load(original_path)
        
        # Analyze point cloud statistics
        completed_center = completed.mean(axis=0)
        completed_std = completed.std(axis=0)
        original_center = original.mean(axis=0)
        original_std = original.std(axis=0)
        
        # Check if points are collapsing to center
        spread_ratio = completed_std.mean() / max(original_std.mean(), 1e-6)
        center_distance = np.linalg.norm(completed_center - original_center)
        
        # Simple chamfer distance approximation
        from scipy.spatial import distance_matrix
        if len(completed) > 1000:
            idx = np.random.choice(len(completed), 1000, replace=False)
            completed_sample = completed[idx]
        else:
            completed_sample = completed
            
        if len(original) > 1000:
            idx = np.random.choice(len(original), 1000, replace=False)
            original_sample = original[idx]
        else:
            original_sample = original
        
        dist_matrix = distance_matrix(completed_sample, original_sample)
        chamfer = (dist_matrix.min(axis=1).mean() + dist_matrix.min(axis=0).mean()) / 2
        
        metrics.append({
            'epoch': epoch_num,
            'spread_ratio': spread_ratio,
            'center_distance': center_distance,
            'chamfer': chamfer,
            'completed_std': completed_std.mean(),
            'original_std': original_std.mean()
        })
    
    return metrics

def plot_analysis(log_metrics, vis_metrics):
    """Create diagnostic plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Training curves
    if log_metrics:
        ax = axes[0, 0]
        ax.plot(log_metrics['epochs'], log_metrics['train_loss'], 'b-', label='Train')
        ax.plot(log_metrics['epochs'], log_metrics['val_loss'], 'r-', label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True)
        
        # Learning rate
        ax = axes[0, 1]
        if log_metrics['learning_rate']:
            ax.plot(log_metrics['epochs'][:len(log_metrics['learning_rate'])], 
                   log_metrics['learning_rate'], 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True)
    
    # Visualization metrics
    if vis_metrics:
        epochs = [m['epoch'] for m in vis_metrics]
        
        ax = axes[0, 2]
        chamfer = [m['chamfer'] for m in vis_metrics]
        ax.plot(epochs, chamfer, 'o-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Chamfer Distance')
        ax.set_title('Point Cloud Quality')
        ax.grid(True)
        
        ax = axes[1, 0]
        spread = [m['spread_ratio'] for m in vis_metrics]
        ax.plot(epochs, spread, 'o-')
        ax.axhline(y=1.0, color='r', linestyle='--', label='Target')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spread Ratio')
        ax.set_title('Output Spread vs Original\n(< 1 = collapsed)')
        ax.legend()
        ax.grid(True)
        
        ax = axes[1, 1]
        center_dist = [m['center_distance'] for m in vis_metrics]
        ax.plot(epochs, center_dist, 'o-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Center Distance')
        ax.set_title('Center Point Offset')
        ax.grid(True)
        
        ax = axes[1, 2]
        completed_std = [m['completed_std'] for m in vis_metrics]
        original_std = [m['original_std'] for m in vis_metrics]
        ax.plot(epochs, completed_std, 'b-', label='Completed')
        ax.plot(epochs, original_std, 'r--', label='Original')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Std Dev')
        ax.set_title('Point Cloud Spread')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150)
    print("\nSaved analysis plots to training_analysis.png")
    plt.show()

def diagnose_training(log_metrics, vis_metrics):
    """Diagnose training issues and provide recommendations"""
    print("\n" + "="*60)
    print("TRAINING DIAGNOSIS")
    print("="*60)
    
    issues = []
    recommendations = []
    
    # Check if loss is decreasing
    if log_metrics:
        final_train = log_metrics['train_loss'][-1]
        initial_train = log_metrics['train_loss'][0]
        
        if final_train > initial_train * 0.9:
            issues.append("❌ Training loss not decreasing significantly")
            recommendations.append("• Lower learning rate (try 1e-5 or 5e-6)")
            recommendations.append("• Check if model architecture is too simple")
        else:
            print("✓ Training loss decreasing properly")
        
        # Check for overfitting
        if len(log_metrics['val_loss']) > 20:
            recent_val = np.mean(log_metrics['val_loss'][-10:])
            mid_val = np.mean(log_metrics['val_loss'][len(log_metrics['val_loss'])//2:len(log_metrics['val_loss'])//2+10])
            if recent_val > mid_val * 1.1:
                issues.append("⚠️  Validation loss increasing (overfitting)")
                recommendations.append("• Add dropout or weight decay")
                recommendations.append("• Reduce model capacity")
    
    # Check visualization metrics
    if vis_metrics:
        latest = vis_metrics[-1]
        
        # Check if points are collapsing
        if latest['spread_ratio'] < 0.5:
            issues.append("❌ Output points collapsing to center (spread ratio: {:.2f})".format(latest['spread_ratio']))
            recommendations.append("• Model architecture issue - decoder too weak")
            recommendations.append("• Add repulsion loss with higher weight")
            recommendations.append("• Use skip connections from encoder")
        elif latest['spread_ratio'] < 0.8:
            issues.append("⚠️  Output points partially collapsed (spread ratio: {:.2f})".format(latest['spread_ratio']))
            recommendations.append("• Increase model capacity")
            recommendations.append("• Train for more epochs with lower learning rate")
        else:
            print("✓ Output spread looks reasonable")
        
        # Check chamfer distance improvement
        if len(vis_metrics) > 10:
            early_chamfer = np.mean([m['chamfer'] for m in vis_metrics[:5]])
            recent_chamfer = np.mean([m['chamfer'] for m in vis_metrics[-5:]])
            
            if recent_chamfer > early_chamfer * 0.95:
                issues.append("❌ Chamfer distance not improving")
                recommendations.append("• Architecture may be inadequate")
                recommendations.append("• Try enhanced model architecture")
            else:
                improvement = (1 - recent_chamfer/early_chamfer) * 100
                print(f"✓ Chamfer distance improved by {improvement:.1f}%")
    
    # Print diagnosis
    if issues:
        print("\n🔍 ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("\n✅ Training appears to be progressing well!")
    
    # Architecture recommendation
    if vis_metrics and vis_metrics[-1]['spread_ratio'] < 0.6:
        print("\n" + "="*60)
        print("🚨 CRITICAL: Model Architecture Inadequate")
        print("="*60)
        print("The current model (FullModelSnowflake) is producing collapsed outputs.")
        print("This indicates fundamental architecture limitations.\n")
        print("STRONGLY RECOMMENDED:")
        print("1. Switch to the enhanced architecture (EnhancedPointCompletionModel)")
        print("2. Start training from scratch with better initialization")
        print("3. Use the improved loss function (CombinedLossWithEMD)")
        print("\nThe enhanced model includes:")
        print("  • Folding-based decoder (better than random seed points)")
        print("  • Attention-based feature propagation")
        print("  • Skip connections from encoder")
        print("  • More gradual upsampling (512->1024->2048->4096->8192)")
        print("  • Multiple loss terms to prevent collapse")
    
    return len(issues) > 0

def main():
    parser = argparse.ArgumentParser(description="Analyze GCNN training progress")
    parser.add_argument("--log_dir", default="logs", help="Log directory")
    parser.add_argument("--vis_dir", default="visuals", help="Visualizations directory")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    print("Analyzing training progress...")
    
    # Load metrics
    log_metrics = load_training_logs(args.log_dir)
    vis_metrics = analyze_visualizations(args.vis_dir)
    
    # Create plots
    if log_metrics or vis_metrics:
        plot_analysis(log_metrics, vis_metrics)
    
    # Diagnose issues
    has_issues = diagnose_training(log_metrics, vis_metrics)
    
    # Check latest checkpoint
    latest_checkpoint = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        print(f"\n📦 Latest checkpoint: Epoch {checkpoint.get('epoch', 'unknown')}")
        
        if has_issues:
            print("\n" + "="*60)
            print("NEXT STEPS:")
            print("="*60)
            print("Option 1: Continue with current model (NOT RECOMMENDED)")
            print("  python train_distributed.py --resume --checkpoint_path checkpoints/latest_checkpoint.pth \\")
            print("    --learning_rate 5e-6 --num_epochs 200")
            print("\nOption 2: Switch to enhanced model (RECOMMENDED)")
            print("  python train_enhanced.py --dataset_root data/S3DIS \\")
            print("    --num_epochs 200 --batch_size 2")
    
    print("\n" + "="*60)
    print("Analysis complete!")

if __name__ == "__main__":
    main()