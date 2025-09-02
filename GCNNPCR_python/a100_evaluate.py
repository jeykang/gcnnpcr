#!/usr/bin/env python
"""
Evaluation script for trained point completion models
Computes metrics and generates visualizations
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import dataset and model
from minimal_main_4 import S3DISDataset, chamfer_distance


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine which model architecture to use
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'model_type'):
        model_type = checkpoint['args'].model_type
    else:
        # Try to infer from state dict keys
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Recognise multi‑GPU models by looking for gconvs/fusion/feat_processor keys.
        if any(k.startswith('encoder.encoders.0.gconvs') for k in state_dict) or \
           any(k.startswith('encoder.fusion') for k in state_dict) or \
           any(k.startswith('feat_processor') for k in state_dict) or \
           'encoder.encoders.0.weight' in state_dict or \
           'encoder.encoders.0.conv_cls.weight' in state_dict:
            model_type = 'multigpu'
        else:
            model_type = 'original'
    
    print(f"Loading {model_type} model architecture...")
    
    if model_type == 'multigpu':
        try:
            from multigpu_enhanced_model import create_multigpu_model
            model = create_multigpu_model()
        except ImportError:
            from a100_model import create_multigpu_model
            model = create_multigpu_model()
    else:
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
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def evaluate_model(model, dataloader, device='cuda', max_batches=None):
    """Evaluate model on dataset"""
    model.eval()
    
    metrics = {
        'chamfer': [],
        'coverage': [],
        'spread': [],
        'density_ratio': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and i >= max_batches:
                break
            
            partial = batch['partial'].to(device)
            full = batch['full'].to(device)
            
            # Forward pass
            output = model(partial)  # [B, 3, N]
            output = output.permute(0, 2, 1)  # [B, N, 3]
            
            gt_coords = full[..., :3]
            partial_coords = partial[..., :3]
            
            # Compute metrics for each sample in batch
            B = output.shape[0]
            for b in range(B):
                pred = output[b]
                gt = gt_coords[b]
                partial_b = partial_coords[b]
                
                # Filter out masked points from partial
                mask = partial_b.abs().sum(dim=-1) > 1e-6
                partial_valid = partial_b[mask] if mask.any() else partial_b
                
                # Chamfer distance
                cd = compute_chamfer_distance(pred.unsqueeze(0), gt.unsqueeze(0))
                metrics['chamfer'].append(cd.item())
                
                # Coverage: how well does output cover the partial input
                if len(partial_valid) > 0:
                    dist_to_partial = torch.cdist(partial_valid.unsqueeze(0), pred.unsqueeze(0))
                    coverage = dist_to_partial.min(dim=-1)[0].mean()
                    metrics['coverage'].append(coverage.item())
                
                # Spread: standard deviation of points
                pred_std = pred.std(dim=0).mean()
                gt_std = gt.std(dim=0).mean()
                spread_ratio = pred_std / (gt_std + 1e-8)
                metrics['spread'].append(spread_ratio.item())
                
                # Density: average distance to nearest neighbor
                pred_density = compute_density(pred)
                gt_density = compute_density(gt)
                density_ratio = pred_density / (gt_density + 1e-8)
                metrics['density_ratio'].append(density_ratio.item())
    
    # Compute mean and std for each metric
    results = {}
    for key, values in metrics.items():
        values = np.array(values)
        results[f'{key}_mean'] = values.mean()
        results[f'{key}_std'] = values.std()
    
    return results


def compute_chamfer_distance(pred, gt):
    """Compute Chamfer distance between predicted and ground truth"""
    dist1 = torch.cdist(pred, gt).min(dim=-1)[0].mean()
    dist2 = torch.cdist(gt, pred).min(dim=-1)[0].mean()
    return (dist1 + dist2) / 2


def compute_density(points, k=10):
    """Compute average distance to k nearest neighbors"""
    dist = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    dist = dist + torch.eye(len(points), device=points.device) * 1e10
    knn_dist, _ = dist.topk(k, largest=False)
    return knn_dist.mean()


def visualize_samples(model, dataloader, output_dir, num_samples=5, device='cuda'):
    """Generate visualization of sample completions"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    samples_generated = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_generated >= num_samples:
                break
            
            partial = batch['partial'].to(device)
            full = batch['full'].to(device)
            
            # Forward pass
            output = model(partial)
            output = output.permute(0, 2, 1)
            
            # Process each sample in batch
            B = min(output.shape[0], num_samples - samples_generated)
            for b in range(B):
                partial_b = partial[b, ..., :3].cpu().numpy()
                output_b = output[b].cpu().numpy()
                gt_b = full[b, ..., :3].cpu().numpy()
                
                # Filter masked points
                mask = np.abs(partial_b).sum(axis=-1) > 1e-6
                partial_filtered = partial_b[mask]
                
                # Create figure with 3 subplots
                fig = plt.figure(figsize=(15, 5))
                
                # Partial input
                ax1 = fig.add_subplot(131, projection='3d')
                ax1.scatter(partial_filtered[:, 0], 
                          partial_filtered[:, 1], 
                          partial_filtered[:, 2], 
                          c='red', s=1, alpha=0.5)
                ax1.set_title('Partial Input')
                ax1.set_box_aspect([1,1,1])
                
                # Completed output
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(output_b[:, 0], 
                          output_b[:, 1], 
                          output_b[:, 2], 
                          c='blue', s=1, alpha=0.5)
                ax2.set_title('Completed Output')
                ax2.set_box_aspect([1,1,1])
                
                # Ground truth
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(gt_b[:, 0], 
                          gt_b[:, 1], 
                          gt_b[:, 2], 
                          c='green', s=1, alpha=0.5)
                ax3.set_title('Ground Truth')
                ax3.set_box_aspect([1,1,1])
                
                # Set consistent axis limits
                all_points = np.concatenate([partial_filtered, output_b, gt_b], axis=0)
                min_vals = all_points.min(axis=0)
                max_vals = all_points.max(axis=0)
                
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlim(min_vals[0], max_vals[0])
                    ax.set_ylim(min_vals[1], max_vals[1])
                    ax.set_zlim(min_vals[2], max_vals[2])
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, f'sample_{samples_generated:03d}.png')
                plt.savefig(save_path, dpi=150)
                plt.close()
                
                # Also save as numpy arrays for further analysis
                np_dir = os.path.join(output_dir, f'sample_{samples_generated:03d}')
                os.makedirs(np_dir, exist_ok=True)
                np.save(os.path.join(np_dir, 'partial.npy'), partial_filtered)
                np.save(os.path.join(np_dir, 'output.npy'), output_b)
                np.save(os.path.join(np_dir, 'ground_truth.npy'), gt_b)
                
                samples_generated += 1
                
                print(f"Saved visualization {samples_generated}/{num_samples}")


def generate_report(results, output_path):
    """Generate evaluation report"""
    report = []
    report.append("="*60)
    report.append("POINT CLOUD COMPLETION EVALUATION REPORT")
    report.append("="*60)
    report.append("")
    
    # Metrics
    report.append("METRICS:")
    report.append("-"*40)
    report.append(f"Chamfer Distance:     {results['chamfer_mean']:.4f} ± {results['chamfer_std']:.4f}")
    report.append(f"Coverage Distance:    {results['coverage_mean']:.4f} ± {results['coverage_std']:.4f}")
    report.append(f"Spread Ratio:        {results['spread_mean']:.4f} ± {results['spread_std']:.4f}")
    report.append(f"Density Ratio:       {results['density_ratio_mean']:.4f} ± {results['density_ratio_std']:.4f}")
    report.append("")
    
    # Quality assessment
    report.append("QUALITY ASSESSMENT:")
    report.append("-"*40)
    
    if results['spread_mean'] < 0.5:
        report.append("⚠️  WARNING: Low spread ratio indicates point collapse!")
        report.append("   The model is generating clustered outputs.")
    elif results['spread_mean'] < 0.8:
        report.append("⚠️  CAUTION: Moderate spread ratio.")
        report.append("   Some clustering may be present.")
    else:
        report.append("✓  Good spread ratio - points are well distributed.")
    
    if results['coverage_mean'] > 0.05:
        report.append("⚠️  WARNING: High coverage distance!")
        report.append("   The model is not covering the input well.")
    else:
        report.append("✓  Good coverage of input points.")
    
    if abs(results['density_ratio_mean'] - 1.0) > 0.3:
        report.append("⚠️  WARNING: Density mismatch with ground truth.")
    else:
        report.append("✓  Good density match with ground truth.")
    
    report.append("")
    report.append("="*60)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate point completion model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to S3DIS dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--max_batches', type=int, default=None, help='Max batches to evaluate')
    parser.add_argument('--num_vis_samples', type=int, default=10, help='Number of visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    
    # Create dataset
    print(f"Loading dataset from {args.dataset_root}...")
    val_dataset = S3DISDataset(
        root=args.dataset_root,
        mask_ratio=0.4,
        num_points=8192,
        split="val",
        normal_k=16,
        patches_per_room=2,
        train_areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        test_areas=["Area_6"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, val_loader, device=args.device, 
                           max_batches=args.max_batches)
    
    # Generate visualizations
    print(f"Generating {args.num_vis_samples} visualizations...")
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    visualize_samples(model, val_loader, vis_dir, 
                     num_samples=args.num_vis_samples, device=args.device)
    
    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    generate_report(results, report_path)
    
    # Save metrics as JSON
    import json
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")
    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Visualizations: {vis_dir}")


if __name__ == '__main__':
    main()