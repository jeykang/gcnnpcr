#!/bin/bash
#SBATCH --job-name=gcnn_analyze
#SBATCH --output=logs/analyze_%A.out
#SBATCH --error=logs/analyze_%A.err
#SBATCH --nodes=1                # Number of nodes (change for multi-node)
#SBATCH --ntasks-per-node=8      # Number of GPUs per node
#SBATCH --gres=gpu:8             # Request 8 GPUs
#SBATCH --time=48:00:00          # Time limit
#SBATCH --nodelist=hpc-pr-a-pod09

# ============================================================================
#  GCNN Training Analysis - Diagnose Training Issues
# ============================================================================

echo "=============================================="
echo "GCNN Training Analysis"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "=============================================="

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-gcnnpcr_alt.sif}"
LOG_DIR="${PROJECT_ROOT}/logs"
VIS_DIR="${PROJECT_ROOT}/visuals"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"

# Verify container exists
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity container not found: $SINGULARITY_IMAGE"
    echo "Please build it first or set SINGULARITY_IMAGE variable"
    exit 1
fi

# Verify directories exist
if [ ! -d "$LOG_DIR" ]; then
    echo "WARNING: Log directory not found: $LOG_DIR"
fi

if [ ! -d "$VIS_DIR" ]; then
    echo "WARNING: Visualization directory not found: $VIS_DIR"
fi

echo -e "\nAnalyzing training from:"
echo "  Logs: $LOG_DIR"
echo "  Visualizations: $VIS_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo ""

# Run analysis inside container
singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    "$SINGULARITY_IMAGE" \
    python /workspace/analyze_training.py \
        --log_dir /workspace/logs \
        --vis_dir /workspace/visuals \
        --checkpoint_dir /workspace/checkpoints

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n=============================================="
    echo "Analysis completed successfully!"
    echo "Check training_analysis.png for diagnostic plots"
    echo "=============================================="
    
    # Copy analysis results to a timestamped file
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [ -f "${PROJECT_ROOT}/training_analysis.png" ]; then
        cp "${PROJECT_ROOT}/training_analysis.png" "${PROJECT_ROOT}/analysis_${TIMESTAMP}.png"
        echo "Analysis saved to: analysis_${TIMESTAMP}.png"
    fi
else
    echo -e "\n=============================================="
    echo "Analysis failed with exit code $EXIT_CODE"
    echo "Check the error log above for details"
    echo "=============================================="
fi

exit $EXIT_CODE