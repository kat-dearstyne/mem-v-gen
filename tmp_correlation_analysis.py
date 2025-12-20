#!/usr/bin/env python3
"""
Temporary script to analyze correlations between error metrics and jaccard index.
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load the data
error_df = pd.read_csv("output/memorized-clt/memorized vs. random/error-results-individual.csv")
combined_df = pd.read_csv("output/memorized-clt/final/combined_results.csv")

# Filter combined_df for "memorized vs rephrased" condition
rephrased_df = combined_df[combined_df['prompt_type'] == 'memorized vs rephrased'].copy()
rephrased_df = rephrased_df.set_index('config_name')

# Pivot error_df to have metrics as columns, samples as rows
# We'll look at g1_score, g2_score, and delta for each metric
metrics = error_df['metric'].unique()

# Create a wide dataframe with all error metrics
error_wide = error_df.pivot(index='sample', columns='metric', values=['g1_score', 'g2_score', 'delta'])
# Flatten column names
error_wide.columns = [f"{col[1]}_{col[0]}" for col in error_wide.columns]

# Merge with rephrased jaccard data
merged = error_wide.join(rephrased_df[['jaccard_index', 'relative_jaccard']], how='inner')

# Exclude lorem as potential outlier
merged = merged.drop('lorem', errors='ignore')

print("=" * 70)
print("CORRELATION ANALYSIS: Error Metrics vs Jaccard Index (memorized vs rephrased)")
print("=" * 70)
print(f"\nNumber of samples: {len(merged)}")
print(f"\nSamples: {list(merged.index)}")

# Compute correlations for all error metric columns vs jaccard_index
print("\n" + "-" * 70)
print("Pearson Correlations with Jaccard Index")
print("-" * 70)

correlations = []
for col in error_wide.columns:
    r, p = stats.pearsonr(merged[col], merged['jaccard_index'])
    correlations.append({
        'metric': col,
        'pearson_r': r,
        'p_value': p,
        'significant': p < 0.05
    })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('pearson_r', key=abs, ascending=False)

print(f"\n{'Metric':<35} {'Pearson r':>12} {'p-value':>12} {'Sig.':>6}")
print("-" * 70)
for _, row in corr_df.iterrows():
    sig = "*" if row['significant'] else ""
    print(f"{row['metric']:<35} {row['pearson_r']:>12.4f} {row['p_value']:>12.4f} {sig:>6}")

# Also check Spearman (rank) correlations
print("\n" + "-" * 70)
print("Spearman Correlations with Jaccard Index")
print("-" * 70)

spearman_correlations = []
for col in error_wide.columns:
    r, p = stats.spearmanr(merged[col], merged['jaccard_index'])
    spearman_correlations.append({
        'metric': col,
        'spearman_r': r,
        'p_value': p,
        'significant': p < 0.05
    })

spearman_df = pd.DataFrame(spearman_correlations)
spearman_df = spearman_df.sort_values('spearman_r', key=abs, ascending=False)

print(f"\n{'Metric':<35} {'Spearman r':>12} {'p-value':>12} {'Sig.':>6}")
print("-" * 70)
for _, row in spearman_df.iterrows():
    sig = "*" if row['significant'] else ""
    print(f"{row['metric']:<35} {row['spearman_r']:>12.4f} {row['p_value']:>12.4f} {sig:>6}")

# Focus on the most interesting metrics (g1_score which represents memorized condition)
print("\n" + "=" * 70)
print("SUMMARY: Top correlations (g1_score metrics only)")
print("=" * 70)

g1_metrics = [c for c in error_wide.columns if 'g1_score' in c]
summary_corrs = []
for col in g1_metrics:
    pearson_r, pearson_p = stats.pearsonr(merged[col], merged['jaccard_index'])
    spearman_r, spearman_p = stats.spearmanr(merged[col], merged['jaccard_index'])
    summary_corrs.append({
        'metric': col.replace('_g1_score', ''),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    })

summary_df = pd.DataFrame(summary_corrs)
summary_df = summary_df.sort_values('pearson_r', key=abs, ascending=False)

print(f"\n{'Metric':<20} {'Pearson r':>10} {'(p)':>10} {'Spearman r':>12} {'(p)':>10}")
print("-" * 70)
for _, row in summary_df.iterrows():
    print(f"{row['metric']:<20} {row['pearson_r']:>10.4f} {row['pearson_p']:>10.4f} {row['spearman_r']:>12.4f} {row['spearman_p']:>10.4f}")

# Print the actual data for inspection
print("\n" + "=" * 70)
print("RAW DATA (for inspection)")
print("=" * 70)
print("\nJaccard Index vs Average Precision (g1_score):")
print(merged[['jaccard_index', 'average_precision_g1_score']].sort_values('jaccard_index', ascending=False).to_string())
