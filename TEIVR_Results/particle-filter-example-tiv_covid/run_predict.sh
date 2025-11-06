#!/bin/bash
#SBATCH --job-name=predict_pp
#SBATCH --output=logs/slurm_predict_%j.out
#SBATCH --error=logs/slurm_predict_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

# Optional: Uncomment to receive email notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tomkimpson@gmail.com

# Print job information
echo "=========================================="
echo "Posterior Predictive Simulation - COVID TIV"
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

echo "pypfilt version:"
python -c "import pypfilt; print(pypfilt.__version__)" 2>/dev/null || echo "pypfilt version not available"
echo ""

# Default configuration (production settings)
POSTERIOR_RUN="results/npe/20250103_existing_primary"
PATIENTS="all"
NUM_SAMPLES=50
NUM_PARTICLES=50
HORIZON=14
SEED=42
NUM_WORKERS=8
OUTPUT_DIR=""  # Will be auto-generated with timestamp if not specified

# Parse command line arguments for flexibility
while [[ $# -gt 0 ]]; do
    case $1 in
        --posterior-run)
            POSTERIOR_RUN="$2"
            shift 2
            ;;
        --patients)
            # Accept comma-separated patient IDs: 432192,443108,444332
            PATIENTS="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --particles)
            NUM_PARTICLES="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --smoke-test)
            # Quick test with minimal parameters
            NUM_SAMPLES=5
            NUM_PARTICLES=100
            HORIZON=3
            PATIENTS="432192"
            OUTPUT_DIR="results/posterior_predictive/smoke_test"
            echo "SMOKE TEST MODE ENABLED"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available options:"
            echo "  --posterior-run <path>      (default: results/npe/20250103_existing_primary)"
            echo "  --patients <list|all>       (default: all, use comma-separated: 432192,443108)"
            echo "  --num-samples <int>         (default: 1000)"
            echo "  --particles <int>           (default: 500)"
            echo "  --horizon <int>             (default: 14)"
            echo "  --seed <int>                (default: 42)"
            echo "  --num-workers <int>         (default: 8)"
            echo "  --output-dir <path>         (default: auto-generated with timestamp)"
            echo "  --smoke-test                (quick test: 5 samples, 100 particles, H=3)"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Posterior run: $POSTERIOR_RUN"
echo "  Patients: $PATIENTS"
echo "  Parameter samples: $NUM_SAMPLES"
echo "  Particles per filter: $NUM_PARTICLES"
echo "  Prediction horizon: $HORIZON days"
echo "  Random seed: $SEED"
echo "  Parallel workers: $NUM_WORKERS"
echo "  Output directory: $([ -z "$OUTPUT_DIR" ] && echo "auto-generated" || echo "$OUTPUT_DIR")"
echo ""

# Check that the posterior exists
if [ ! -d "$POSTERIOR_RUN" ]; then
    echo "ERROR: Posterior run directory not found: $POSTERIOR_RUN"
    echo "Please check that the NPE inference has been completed."
    exit 1
fi

# Check for posterior model
MODEL_FILES=$(find "$POSTERIOR_RUN/models" -name "posterior_N*.pkl" 2>/dev/null | wc -l)
if [ "$MODEL_FILES" -eq 0 ]; then
    echo "ERROR: No posterior model found in: $POSTERIOR_RUN/models/"
    echo "Please run NPE training and inference first."
    exit 1
fi

echo "Found posterior in: $POSTERIOR_RUN"
echo ""

# Build the command
CMD="python -u COVID_TEIVR_Predict.py \
    --posterior-run $POSTERIOR_RUN \
    --num-samples $NUM_SAMPLES \
    --particles $NUM_PARTICLES \
    --horizon $HORIZON \
    --seed $SEED \
    --num-workers $NUM_WORKERS"

# Add patients if not "all"
if [ "$PATIENTS" != "all" ]; then
    # Convert comma-separated to space-separated for Python argparse
    PATIENTS_SPACE="${PATIENTS//,/ }"
    CMD="$CMD --patients $PATIENTS_SPACE"
fi

# Add output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

# Run the prediction
echo "=========================================="
echo "RUNNING POSTERIOR PREDICTIVE SIMULATION"
echo "=========================================="
echo "Command: $CMD"
echo ""

time $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PREDICTION COMPLETED SUCCESSFULLY!"
    echo "=========================================="

    # Find the output directory (either specified or auto-generated)
    if [ -z "$OUTPUT_DIR" ]; then
        # Find the most recent output directory
        OUTPUT_DIR=$(ls -td results/posterior_predictive/*/ 2>/dev/null | head -1 | sed 's:/$::')
    fi

    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "Results saved to: $OUTPUT_DIR"
        echo ""
        echo "Output summary:"
        echo "  Patient directories: $(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | wc -l)"

        # Validate outputs
        echo ""
        echo "Validating outputs..."
        FAILED=0
        TOTAL=0
        for PATIENT_DIR in "$OUTPUT_DIR"/*/; do
            TOTAL=$((TOTAL + 1))
            PATIENT=$(basename "$PATIENT_DIR")
            if [ ! -f "$PATIENT_DIR/summary_statistics.csv" ]; then
                echo "  WARNING: Missing summary for patient $PATIENT"
                FAILED=$((FAILED + 1))
            elif [ -f "$PATIENT_DIR/filter_trajectory.csv" ]; then
                echo "  ✓ Patient $PATIENT: Complete (with filter trajectory)"
            else
                echo "  ✓ Patient $PATIENT: Complete (no filter trajectory)"
            fi
        done

        if [ $FAILED -gt 0 ]; then
            echo ""
            echo "  $FAILED of $TOTAL patients failed validation"
        else
            echo ""
            echo "  All $TOTAL patients completed successfully"
        fi

        # Show some stats from the first patient if available
        FIRST_PATIENT=$(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | head -1 | xargs basename)
        if [ -n "$FIRST_PATIENT" ]; then
            echo ""
            echo "Example patient: $FIRST_PATIENT"
            if [ -f "$OUTPUT_DIR/$FIRST_PATIENT/timing.txt" ]; then
                echo "Timing information:"
                cat "$OUTPUT_DIR/$FIRST_PATIENT/timing.txt" | sed 's/^/  /'
            fi
        fi
    fi

    echo ""
    echo "End time: $(date)"
else
    echo ""
    echo "=========================================="
    echo "PREDICTION FAILED"
    echo "=========================================="
    echo "Exit code: $?"
    echo "End time: $(date)"
    exit 1
fi
