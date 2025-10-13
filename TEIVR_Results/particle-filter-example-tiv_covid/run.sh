#!/bin/bash
#SBATCH --job-name=covid_tiv_inference
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=24:00:00

# Optional: Uncomment to receive email notifications
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@example.com

# Print job information
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

# Run the particle filter inference
echo "Starting COVID TIV inference..."
time python COVID_TEIVR_Inf_loop.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Job completed successfully!"
else
    echo ""
    echo "Job failed with exit code $?"
fi

echo "End time: $(date)"
