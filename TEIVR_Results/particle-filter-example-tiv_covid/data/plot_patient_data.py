#!/usr/bin/env python3
"""
Plot observed patient viral load data.

This script creates individual plots for each patient showing:
- X-axis: Time (days post-infection)
- Y-axis: Viral load (copies/mL) on logarithmic scale
- Title: Patient ID XXX

Figures are saved in the images/ subdirectory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scienceplots

# Apply science plots style
plt.style.use('science')

# Patient IDs to process
PATIENT_IDS = ['432192', '443108', '444332', '444391', '445602', '451152']

# Set up directories
data_dir = Path(__file__).parent
images_dir = data_dir / 'images'
images_dir.mkdir(exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Images will be saved to: {images_dir}")
print(f"\nProcessing {len(PATIENT_IDS)} patients...\n")

# Process each patient
for patient_id in PATIENT_IDS:
    print(f"Processing Patient ID {patient_id}...")

    # Load patient data
    data_file = data_dir / f"{patient_id}.ssv"
    df = pd.read_csv(data_file, sep=' ')

    # Extract time and viral load data
    time_days = df['time'].values
    log10_viral_load = df['value'].values

    # Convert from log10 space to linear scale
    viral_load = 10 ** log10_viral_load

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot data
    ax.plot(time_days, viral_load, 'o-', markersize=6, linewidth=1.5)

    # Set logarithmic scale for y-axis
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Viral load (copies/mL)')
    ax.set_title(f'Patient ID {patient_id}')

    # Grid for better readability
    ax.grid(True, alpha=0.3)

    # Save figure
    output_file = images_dir / f"{patient_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

print(f"\nAll plots completed! Check {images_dir} for results.")
