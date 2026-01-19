#!/bin/bash
#SBATCH --job-name=covid_npe_gpu
#SBATCH --output=logs/slurm_npe_gpu_%j.out
#SBATCH --error=logs/slurm_npe_gpu_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --partition=milan-c
#SBATCH --gres=gpu:1

# Optional: Uncomment to receive email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tomkimpson@gmail.com

# Print job information
echo "=========================================="
echo "Neural Posterior Estimation - COVID TIV (GPU)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Activate virtual environment
source venv/bin/activate

# Print Python and package versions for reproducibility
echo "Python version:"
python --version
echo ""

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "PyTorch not installed"
echo ""

echo "PyTorch CUDA available:"
python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Cannot check CUDA"
echo ""

echo "PyTorch CUDA device:"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" 2>/dev/null || echo "Cannot check device"
echo ""

echo "SBI version:"
python -c "import sbi; print(sbi.__version__)" 2>/dev/null || echo "SBI not installed"
echo ""

# Configuration
NUM_TRAJECTORIES=10000
NUM_TIMEPOINTS=10
NUM_WORKERS=8
PATIENTS="all"
NUM_SAMPLES=10000

# Create timestamped output directory (shared across all stages)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/npe/${RUN_TIMESTAMP}_gpu"

# Parse command line arguments for flexibility
SKIP_SIMULATE=false
SKIP_TRAIN=false
SKIP_INFER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-trajectories)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --skip-simulate)
            SKIP_SIMULATE=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-infer)
            SKIP_INFER=true
            shift
            ;;
        --patients)
            PATIENTS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Number of trajectories: $NUM_TRAJECTORIES"
echo "  Number of timepoints: $NUM_TIMEPOINTS"
echo "  Number of workers: $NUM_WORKERS"
echo "  Patients: $PATIENTS"
echo "  Posterior samples: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo "  Skip simulate: $SKIP_SIMULATE"
echo "  Skip train: $SKIP_TRAIN"
echo "  Skip infer: $SKIP_INFER"
echo ""

# Stage 1: Simulate training data
if [ "$SKIP_SIMULATE" = false ]; then
    echo "=========================================="
    echo "STAGE 1: SIMULATE"
    echo "=========================================="
    time python -u COVID_TEIVR_NPE.py simulate \
        --num-trajectories $NUM_TRAJECTORIES \
        --num-timepoints $NUM_TIMEPOINTS \
        --num-workers $NUM_WORKERS \
        --output-dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo ""
        echo "Stage 1 completed successfully!"
    else
        echo ""
        echo "Stage 1 failed with exit code $?"
        exit 1
    fi
else
    echo "Skipping Stage 1 (simulate)"
fi

echo ""

# Stage 2: Train neural posterior estimator
if [ "$SKIP_TRAIN" = false ]; then
    echo "=========================================="
    echo "STAGE 2: TRAIN (GPU-ACCELERATED)"
    echo "=========================================="
    time python -u COVID_TEIVR_NPE.py train \
        --num-trajectories $NUM_TRAJECTORIES \
        --output-dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo ""
        echo "Stage 2 completed successfully!"
    else
        echo ""
        echo "Stage 2 failed with exit code $?"
        exit 1
    fi
else
    echo "Skipping Stage 2 (train)"
fi

echo ""

# Stage 3: Inference on patient data
if [ "$SKIP_INFER" = false ]; then
    echo "=========================================="
    echo "STAGE 3: INFER (GPU-ACCELERATED)"
    echo "=========================================="
    time python -u COVID_TEIVR_NPE.py infer \
        --patients $PATIENTS \
        --num-trajectories $NUM_TRAJECTORIES \
        --num-samples $NUM_SAMPLES \
        --num-timepoints $NUM_TIMEPOINTS \
        --output-dir $OUTPUT_DIR \
        --plot

    if [ $? -eq 0 ]; then
        echo ""
        echo "Stage 3 completed successfully!"
    else
        echo ""
        echo "Stage 3 failed with exit code $?"
        exit 1
    fi
else
    echo "Skipping Stage 3 (infer)"
fi

echo ""
echo "=========================================="
echo "ALL STAGES COMPLETED"
echo "=========================================="
echo "End time: $(date)"
