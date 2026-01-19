#!/bin/bash
#SBATCH --job-name=npe_infer
#SBATCH --output=logs/slurm_npe_infer_%j.out
#SBATCH --error=logs/slurm_npe_infer_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00

# Optional: Uncomment to receive email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tomkimpson@gmail.com

# Print job information
echo "=========================================="
echo "NPE Inference Only - COVID TIV"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
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

echo "SBI version:"
python -c "import sbi; print(sbi.__version__)" 2>/dev/null || echo "SBI not installed"
echo ""

# Configuration - using existing results
OUTPUT_DIR="results/npe/20251103_182129"
NUM_TRAJECTORIES=10000
NUM_TIMEPOINTS=10
PATIENTS="all"
NUM_SAMPLES=10000
PLOT_FLAG="--plot"

# Parse command line arguments for flexibility
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-trajectories)
            NUM_TRAJECTORIES="$2"
            shift 2
            ;;
        --patients)
            PATIENTS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-timepoints)
            NUM_TIMEPOINTS="$2"
            shift 2
            ;;
        --no-plot)
            PLOT_FLAG=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available options:"
            echo "  --output-dir <path>         (default: results/npe/20251103_182129)"
            echo "  --num-trajectories <int>    (default: 10000)"
            echo "  --patients <list|all>       (default: all)"
            echo "  --num-samples <int>         (default: 10000)"
            echo "  --num-timepoints <int>      (default: 10)"
            echo "  --no-plot                   (disable plotting)"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of trajectories (trained): $NUM_TRAJECTORIES"
echo "  Number of timepoints: $NUM_TIMEPOINTS"
echo "  Patients: $PATIENTS"
echo "  Posterior samples: $NUM_SAMPLES"
echo "  Generate plots: $([ -z "$PLOT_FLAG" ] && echo "No" || echo "Yes")"
echo ""

# Check that the required model file exists
MODEL_PATH="${OUTPUT_DIR}/models/posterior_N${NUM_TRAJECTORIES}.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Trained model not found at: $MODEL_PATH"
    echo "Please check:"
    echo "  1. Output directory is correct"
    echo "  2. Training was completed successfully"
    echo "  3. num-trajectories matches the training run"
    exit 1
fi

echo "Found trained model: $MODEL_PATH"
echo ""

# Stage 3: Inference on patient data
echo "=========================================="
echo "RUNNING INFERENCE"
echo "=========================================="
time python -u COVID_TEIVR_NPE.py infer \
    --patients $PATIENTS \
    --num-trajectories $NUM_TRAJECTORIES \
    --num-samples $NUM_SAMPLES \
    --num-timepoints $NUM_TIMEPOINTS \
    --output-dir $OUTPUT_DIR \
    $PLOT_FLAG

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "INFERENCE COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results saved to: ${OUTPUT_DIR}/inference/"
    echo "End time: $(date)"
else
    echo ""
    echo "=========================================="
    echo "INFERENCE FAILED"
    echo "=========================================="
    echo "Exit code: $?"
    echo "End time: $(date)"
    exit 1
fi
