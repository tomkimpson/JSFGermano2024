#!/usr/bin/env python3
"""
Script to extract and plot posterior distributions from pypfilt particle filter results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File path
pkl_file = Path(__file__).parent / 'outputs4/432192/src.tiv.RefractoryCellModel_JSF_6000/fit_result.pkl'

# Load the pickle file
print("Loading pickle file...")
with open(pkl_file, 'rb') as f:
    results = pickle.load(f)

print("Results loaded successfully!")

# Extract estimation results
est = results.estimation

# Get the particle snapshot table
snapshot_table = est.tables['snapshot']
print(f"\nExtracted {len(snapshot_table)} particle snapshots")

# Get unique times
times = np.unique(snapshot_table['time'])
print(f"Time points: {times}")

# Use the final time point
final_time = times[-1]
print(f"Using final time point: {final_time}")

# Extract particles from the final time
particles = snapshot_table[snapshot_table['time'] == final_time]
print(f"Number of particles at final time: {len(particles)}")

# Extract weights
weights = particles['weight']
print(f"Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
print(f"Weights sum to: {weights.sum():.6f}")

# Normalize weights to sum to 1 (should already be, but just in case)
weights = weights / weights.sum()

# Parameters to plot (the estimated parameters with non-constant priors)
estimated_params = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']

# Create posterior plots
n_params = len(estimated_params)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Store summary statistics
summary_stats = {}

for i, param in enumerate(estimated_params):
    ax = axes[i]

    # Extract parameter values from particles
    param_values = particles[param]

    # Create weighted histogram
    # Using 50 bins for smooth distribution
    n_bins = 50

    # Calculate histogram with weights
    counts, bins, patches = ax.hist(param_values, bins=n_bins, weights=weights,
                                     alpha=0.7, edgecolor='black', color='steelblue',
                                     density=True)

    # Calculate statistics using weighted particles
    mean_val = np.average(param_values, weights=weights)

    # Weighted variance
    variance = np.average((param_values - mean_val)**2, weights=weights)
    std_val = np.sqrt(variance)

    # Weighted percentiles
    sorted_idx = np.argsort(param_values)
    sorted_vals = param_values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)

    # Find percentiles
    q025 = sorted_vals[np.searchsorted(cumsum, 0.025)]
    q25 = sorted_vals[np.searchsorted(cumsum, 0.25)]
    q50 = sorted_vals[np.searchsorted(cumsum, 0.50)]
    q75 = sorted_vals[np.searchsorted(cumsum, 0.75)]
    q975 = sorted_vals[np.searchsorted(cumsum, 0.975)]

    # Add vertical lines for key statistics
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(q50, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {q50:.3f}')

    # Shade 95% credible interval
    ax.axvspan(q025, q975, alpha=0.2, color='green',
               label=f'95% CI: [{q025:.2f}, {q975:.2f}]')

    ax.set_xlabel(param, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Posterior Distribution: {param}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Store summary stats
    summary_stats[param] = {
        'mean': mean_val,
        'std': std_val,
        'median': q50,
        'q025': q025,
        'q25': q25,
        'q75': q75,
        'q975': q975
    }

plt.tight_layout()

# Save the figure
output_file = pkl_file.parent / 'posterior_distributions.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPosterior distribution plot saved to: {output_file}")

# Print summary statistics
print("\n" + "="*70)
print("POSTERIOR SUMMARY STATISTICS")
print("="*70)
for param in estimated_params:
    if param in summary_stats:
        stats = summary_stats[param]
        print(f"\n{param}:")
        print(f"  Mean:    {stats['mean']:10.6f}")
        print(f"  Std Dev: {stats['std']:10.6f}")
        print(f"  Median:  {stats['median']:10.6f}")
        print(f"  95% CI:  [{stats['q025']:10.6f}, {stats['q975']:10.6f}]")
        print(f"  IQR:     [{stats['q25']:10.6f}, {stats['q75']:10.6f}]")

print("\n" + "="*70)

# Also check covariance table for parameter relationships
print("\n" + "="*70)
print("PARAMETER COVARIANCES (at final time)")
print("="*70)

covar_table = est.tables['covariencetable']
final_covar = covar_table[covar_table['time'] == final_time]

# Show covariances between estimated parameters
print("\nCovariances between parameters:")
for i, p1 in enumerate(estimated_params):
    for p2 in estimated_params[i+1:]:
        mask = ((final_covar['param1'] == p1) & (final_covar['param2'] == p2)) | \
               ((final_covar['param1'] == p2) & (final_covar['param2'] == p1))
        if np.any(mask):
            covar = final_covar[mask]['covar'][0]
            sign = "+" if covar >= 0 else "-"
            print(f"  {p1:6s} <-> {p2:6s}: {sign} {abs(covar):10.6f}")

print("\n" + "="*70)
print("\nDone!")
