#!/bin/bash
#SBATCH --job-name=gcnnpcr_training
#SBATCH --output=logs/gcnnpcr_%A_%a.out
#SBATCH --error=logs/gcnnpcr_%A_%a.err
#SBATCH --nodes=1                # Number of nodes (change for multi-node)
#SBATCH --ntasks-per-node=8      # Number of GPUs per node
#SBATCH --gres=gpu:8             # Request 8 GPUs
#SBATCH --time=48:00:00          # Time limit
#SBATCH --nodelist=hpc-pr-a-pod09

# ============================================================================
#  GCNN Point Cloud Reconstruction - Distributed Training on SLURM
# ============================================================================

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
SINGULARITY_IMAGE="${PROJECT_ROOT}/gcnnpcr.sif"
DATASET_ROOT="${PROJECT_ROOT}/data/S3DIS"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
LOG_DIR="${PROJECT_ROOT}/logs"
VIS_DIR="${PROJECT_ROOT}/visuals"

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-2}           # Batch size per GPU
NUM_EPOCHS=${NUM_EPOCHS:-100}         # Number of training epochs
LEARNING_RATE=${LEARNING_RATE:-1e-4}  # Initial learning rate
NUM_WORKERS=${NUM_WORKERS:-4}         # Data loading workers per GPU

# Model configuration
USE_ATTENTION=${USE_ATTENTION:-true}   # Use attention in encoder

# Create necessary directories
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$VIS_DIR"

echo "=============================================="
echo "GCNN Point Cloud Reconstruction Training"
echo "=============================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Node list:        ${SLURM_JOB_NODELIST}"
echo "Number of nodes:  ${SLURM_NNODES}"
echo "Tasks per node:   ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks:      ${SLURM_NTASKS}"
echo "Project root:     ${PROJECT_ROOT}"
echo "Dataset:          ${DATASET_ROOT}"
echo "Checkpoints:      ${CHECKPOINT_DIR}"
echo "Singularity:      ${SINGULARITY_IMAGE}"
echo "=============================================="

# Build Singularity image if it doesn't exist
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Building Singularity image..."
    singularity build --fakeroot "$SINGULARITY_IMAGE" "${PROJECT_ROOT}/gcnnpcr.def"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Singularity image"
        exit 1
    fi
fi

# Verify dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    echo "Please download and extract the S3DIS dataset"
    exit 1
fi

# Set up distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=${SLURM_NTASKS}

# Model-specific arguments
ATTENTION_ARG=""
if [ "$USE_ATTENTION" = "true" ]; then
    ATTENTION_ARG="--use_attention_encoder"
fi

# Resume training arguments
RESUME_ARGS=""
if [ -f "${CHECKPOINT_DIR}/latest_checkpoint.pth" ]; then
    echo "Found checkpoint, will resume training..."
    RESUME_ARGS="--resume --checkpoint_path ${CHECKPOINT_DIR}/latest_checkpoint.pth"
fi

# Function to launch training on each GPU
launch_training() {
    local RANK=$1
    local LOCAL_RANK=$2
    
    export RANK=$RANK
    export LOCAL_RANK=$LOCAL_RANK
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
    
    singularity exec --nv \
        --bind "${PROJECT_ROOT}:/workspace" \
        --bind "${DATASET_ROOT}:/data" \
        --bind "${CHECKPOINT_DIR}:/checkpoints" \
        --bind "${VIS_DIR}:/visuals" \
        "$SINGULARITY_IMAGE" \
        python /workspace/GCNNPCR_python/train_distributed.py \
            --dataset_root /data \
            --checkpoint_dir /checkpoints \
            --vis_dir /visuals \
            --batch_size $BATCH_SIZE \
            --num_epochs $NUM_EPOCHS \
            --learning_rate $LEARNING_RATE \
            --num_workers $NUM_WORKERS \
            --train_areas "Area_1,Area_2,Area_3,Area_4,Area_5" \
            --test_areas "Area_6" \
            --mask_ratio 0.5 \
            --num_points 8192 \
            --patches_per_room 4 \
            --vis_interval 10 \
            --seed 42 \
            $ATTENTION_ARG \
            $RESUME_ARGS \
        2>&1 | tee "${LOG_DIR}/rank_${RANK}.log"
}

# Launch distributed training
if [ ${SLURM_NNODES} -eq 1 ]; then
    # Single-node multi-GPU training
    echo "Starting single-node multi-GPU training with ${SLURM_NTASKS} GPUs..."
    
    for i in $(seq 0 $((SLURM_NTASKS - 1))); do
        launch_training $i $i &
        PIDS[$i]=$!
        echo "Launched Rank $i with PID ${PIDS[$i]}"
        sleep 2
    done
    
    # Wait for all processes to complete
    FAILED=0
    for i in $(seq 0 $((SLURM_NTASKS - 1))); do
        wait ${PIDS[$i]}
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: Rank $i failed with exit code $EXIT_CODE"
            FAILED=$((FAILED + 1))
        else
            echo "Rank $i completed successfully"
        fi
    done
    
    if [ $FAILED -eq 0 ]; then
        echo "Training completed successfully!"
    else
        echo "Training failed on $FAILED rank(s)"
        exit 1
    fi
    
else
    # Multi-node training
    echo "Starting multi-node training across ${SLURM_NNODES} nodes..."
    
    # Use srun for multi-node launch
    srun --ntasks=${SLURM_NTASKS} \
         --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
         singularity exec --nv \
            --bind "${PROJECT_ROOT}:/workspace" \
            --bind "${DATASET_ROOT}:/data" \
            --bind "${CHECKPOINT_DIR}:/checkpoints" \
            --bind "${VIS_DIR}:/visuals" \
            "$SINGULARITY_IMAGE" \
            python /workspace/GCNNPCR_python/train_distributed.py \
                --dataset_root /data \
                --checkpoint_dir /checkpoints \
                --vis_dir /visuals \
                --batch_size $BATCH_SIZE \
                --num_epochs $NUM_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --num_workers $NUM_WORKERS \
                --train_areas "Area_1,Area_2,Area_3,Area_4,Area_5" \
                --test_areas "Area_6" \
                --mask_ratio 0.5 \
                --num_points 8192 \
                --patches_per_room 4 \
                --vis_interval 10 \
                --seed 42 \
                $ATTENTION_ARG \
                $RESUME_ARGS
fi

echo "=============================================="
echo "Job completed at $(date)"
echo "=============================================="