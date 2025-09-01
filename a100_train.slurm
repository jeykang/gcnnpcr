#!/bin/bash
#SBATCH --job-name=gcnn_a100_optimized
#SBATCH --output=logs/optimized_%A.out
#SBATCH --error=logs/optimized_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --nodelist=hpc-pr-a-pod09
#SBATCH --exclusive  # Get exclusive access to the node

# ============================================================================
#  Optimized Training for 8x NVIDIA A100 GPUs
# ============================================================================

echo "=============================================="
echo "Optimized Multi-GPU Training on A100s"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: 8x NVIDIA A100"
echo "Date: $(date)"
echo "=============================================="

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-gcnnpcr.sif}"
DATASET_ROOT="${PROJECT_ROOT}/data/S3DIS"
EXPERIMENT_NAME="a100_optimized_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="${PROJECT_ROOT}/experiments/${EXPERIMENT_NAME}"

# Training hyperparameters optimized for A100s
BATCH_SIZE=16           # Per GPU - A100 can handle much larger batches
GRAD_ACCUMULATION=2     # Effective batch = 16 * 8 * 2 = 256
NUM_EPOCHS=300          # More epochs for better convergence
LEARNING_RATE=2e-4      # Slightly higher LR for larger batch
NUM_WORKERS=8           # More workers for faster data loading

# Create directories
mkdir -p "$SAVE_DIR"/{checkpoints,visuals,logs}
mkdir -p "$PROJECT_ROOT/logs"

# Copy configuration for reproducibility
cat > "$SAVE_DIR/config.txt" << EOF
Experiment: $EXPERIMENT_NAME
Node: $SLURM_NODELIST
GPUs: 8x NVIDIA A100
Batch size per GPU: $BATCH_SIZE
Gradient accumulation: $GRAD_ACCUMULATION
Effective batch size: $((BATCH_SIZE * 8 * GRAD_ACCUMULATION))
Learning rate: $LEARNING_RATE
Epochs: $NUM_EPOCHS
Date: $(date)
EOF

# Check prerequisites
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Building Singularity container..."
    singularity build --fakeroot "$SINGULARITY_IMAGE" "${PROJECT_ROOT}/gcnnpcr.def"
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

# Set environment for optimal A100 performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for A100 (faster training with minimal accuracy loss)
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}

echo -e "\n--- Training Configuration ---"
echo "Effective batch size: $((BATCH_SIZE * 8 * GRAD_ACCUMULATION))"
echo "Total iterations: $((NUM_EPOCHS * 100))"  # Approximate
echo "Mixed precision: Enabled (FP16)"
echo "TF32: Enabled for A100"
echo "Progressive training: Enabled"
echo "------------------------------\n"

# Save the enhanced model files to experiment directory
cp "${PROJECT_ROOT}/GCNNPCR_python/a100_model.py" "$SAVE_DIR/"
cp "${PROJECT_ROOT}/GCNNPCR_python/a100_train.py" "$SAVE_DIR/"

# Launch training with optimized settings
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATASET_ROOT}:/data" \
    --bind "${SAVE_DIR}:/experiment" \
    "$SINGULARITY_IMAGE" \
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=8 \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        /workspace/GCNNPCR_python/a100_train.py \
            --dataset_root /data \
            --checkpoint_dir /experiment/checkpoints \
            --vis_dir /experiment/visuals \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation $GRAD_ACCUMULATION \
            --num_epochs $NUM_EPOCHS \
            --learning_rate $LEARNING_RATE \
            --weight_decay 1e-4 \
            --num_workers $NUM_WORKERS \
            --mask_ratio 0.35 \
            --num_points 8192 \
            --patches_per_room 8 \
            --progressive \
            --start_points 1024 \
            --scheduler onecycle \
            --use_amp \
            --use_emd \
            --chamfer_weight 1.0 \
            --emd_weight 0.3 \
            --repulsion_weight 0.15 \
            --coverage_weight 0.25 \
            --smoothness_weight 0.08 \
            --vis_interval 5 \
            --grad_clip 1.0 \
            2>&1 | tee "$SAVE_DIR/logs/training.log"

EXIT_CODE=$?

echo -e "\n=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved to: $SAVE_DIR"
    
    # Run final evaluation (optional)
    if [ -f "$SAVE_DIR/checkpoints/best_model.pth" ]; then
        echo -e "\nRunning final evaluation..."
        singularity exec --nv \
            --bind "${PROJECT_ROOT}:/workspace" \
            --bind "${DATASET_ROOT}:/data" \
            --bind "${SAVE_DIR}:/experiment" \
            "$SINGULARITY_IMAGE" \
            python /workspace/GCNNPCR_python/a100_evaluate.py \
                --checkpoint /experiment/checkpoints/best_model.pth \
                --dataset_root /data \
                --output_dir /experiment/evaluation \
                --num_vis_samples 10 \
                --max_batches 50
    else
        echo "No best model found for evaluation"
    fi
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check logs at: $SAVE_DIR/logs/training.log"
fi
echo "=============================================="

# Copy logs to experiment directory
cp "${PROJECT_ROOT}/logs/optimized_${SLURM_JOB_ID}.out" "$SAVE_DIR/logs/"
cp "${PROJECT_ROOT}/logs/optimized_${SLURM_JOB_ID}.err" "$SAVE_DIR/logs/"

exit $EXIT_CODE