#!/bin/bash
#SBATCH --job-name=gcnn_enhanced_multi
#SBATCH --output=logs/enhanced_multi_%A.out
#SBATCH --error=logs/enhanced_multi_%A.err
#SBATCH --nodes=1                # Number of nodes (change for multi-node)
#SBATCH --ntasks-per-node=8      # Number of GPUs per node
#SBATCH --gres=gpu:8             # Request 8 GPUs
#SBATCH --time=48:00:00          # Time limit
#SBATCH --nodelist=hpc-pr-a-pod09

# ============================================================================
#  Enhanced Model Multi-GPU Training with Distributed Data Parallel
# ============================================================================

echo "=============================================="
echo "Enhanced Model - Multi-GPU Training"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_NTASKS}"
echo "Date: $(date)"
echo "=============================================="

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-gcnnpcr_alt.sif}"
DATASET_ROOT="${PROJECT_ROOT}/data/S3DIS"
SAVE_DIR="${PROJECT_ROOT}/enhanced_multi"
LOG_DIR="${PROJECT_ROOT}/logs"

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-2}         # Per GPU
NUM_EPOCHS=${NUM_EPOCHS:-200}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
NUM_WORKERS=${NUM_WORKERS:-4}

# Model type (enhanced or original)
USE_ENHANCED=${USE_ENHANCED:-true}

# Create directories
mkdir -p "$SAVE_DIR/checkpoints"
mkdir -p "$SAVE_DIR/visuals"
mkdir -p "$LOG_DIR"

# Check prerequisites
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity container not found: $SINGULARITY_IMAGE"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}

echo -e "\n--- Configuration ---"
echo "Using ${SLURM_NTASKS} GPUs"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * SLURM_NTASKS))"
echo "Model: $([ "$USE_ENHANCED" = "true" ] && echo "Enhanced" || echo "Original")"
echo "Save directory: $SAVE_DIR"
echo "-------------------"

# Create a modified training script that uses the enhanced model
cat > /tmp/train_distributed_enhanced.py << 'EOF'
import sys
sys.path.insert(0, '/workspace/GCNNPCR_python')
sys.path.insert(0, '/workspace')

# Import the distributed training framework
from train_distributed import *

# Override the model creation if using enhanced
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_enhanced_model', action='store_true')
args, remaining = parser.parse_known_args()

if args.use_enhanced_model:
    print("Loading enhanced model architecture...")
    from enhanced_model import EnhancedPointCompletionModel, CombinedLossWithEMD
    
    # Monkey-patch the model creation
    original_main = main
    
    def enhanced_main(args):
        # Temporarily override model creation
        import minimal_main_4
        
        class EnhancedWrapper(EnhancedPointCompletionModel):
            def __init__(self, **kwargs):
                super().__init__(
                    encoder_hidden_dims=[128, 256, 256],
                    encoder_out_dim=256,
                    transformer_dim=256,
                    transformer_heads=8,
                    transformer_layers=6,
                    coarse_points=512,
                    use_attention_encoder=kwargs.get('use_attention_encoder', True)
                )
        
        # Replace the model class
        minimal_main_4.FullModelSnowflake = EnhancedWrapper
        
        # Also enhance the loss function
        original_combined_loss = minimal_main_4.combined_loss
        loss_fn = CombinedLossWithEMD(
            chamfer_weight=1.0,
            repulsion_weight=0.05,
            smoothness_weight=0.01,
            coverage_weight=0.2
        )
        
        def enhanced_loss(pred, gt):
            partial = torch.zeros_like(pred[:, :100, :])  # Dummy partial
            losses = loss_fn(pred, gt, partial)
            return losses['total']
        
        minimal_main_4.combined_loss = enhanced_loss
        
        # Now run original main
        return original_main(args)
    
    main = enhanced_main

# Parse remaining arguments and run
import sys
sys.argv = ['train_distributed.py'] + remaining
from train_distributed import main as train_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Copy all arguments from train_distributed.py
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_areas", type=str, default="")
    parser.add_argument("--test_areas", type=str, default="")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--patches_per_room", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_attention_encoder", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--vis_interval", type=int, default=1)
    parser.add_argument("--vis_dir", type=str, default="visuals")
    parser.add_argument("--log_dir", type=str, default="logs")
    
    args = parser.parse_args()
    train_main(args)
EOF

# Prepare model argument
MODEL_ARG=""
if [ "$USE_ENHANCED" = "true" ]; then
    MODEL_ARG="--use_enhanced_model"
fi

# Launch with torchrun for better distributed support
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATASET_ROOT}:/data" \
    --bind "${SAVE_DIR}/checkpoints:/checkpoints" \
    --bind "${SAVE_DIR}/visuals:/visuals" \
    --bind "${LOG_DIR}:/logs" \
    "$SINGULARITY_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=${SLURM_NTASKS} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        /tmp/train_distributed_enhanced.py \
            $MODEL_ARG \
            --dataset_root /data \
            --checkpoint_dir /checkpoints \
            --vis_dir /visuals \
            --log_dir /logs \
            --batch_size $BATCH_SIZE \
            --num_epochs $NUM_EPOCHS \
            --learning_rate $LEARNING_RATE \
            --num_workers $NUM_WORKERS \
            --mask_ratio 0.4 \
            --num_points 8192 \
            --patches_per_room 4 \
            --vis_interval 5 \
            --seed 42 \
            --use_attention_encoder

EXIT_CODE=$?

echo -e "\n=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Multi-GPU training completed successfully!"
    echo "Results saved to: $SAVE_DIR"
else
    echo "Training failed with exit code $EXIT_CODE"
fi
echo "=============================================="

# Clean up
rm -f /tmp/train_distributed_enhanced.py

exit $EXIT_CODE