# Error Hypothesis Analysis Report

## Overview

This analysis compares base model logits with replacement model logits (using transcoder-reconstructed MLP activations) across three conditions: memorized, made-up, and random. 15 prompts were tested per condition.

## Metrics Computed

| Metric | Description |
|--------|-------------|
| Last Token Cosine | Cosine similarity of logits at final position |
| Cumulative Cosine | Average cosine similarity across all positions |
| Original Accuracy | Fraction of positions where argmax predictions match |
| Top-10 Agreement | Fraction of top-10 tokens that overlap between base and replacement |
| Replacement P(Original Top) | Probability replacement model assigns to original's top token |
| KL Divergence | KL divergence between base and replacement distributions at final position |

## Statistical Analysis

### Significance Testing

**Tests used**: Both parametric and non-parametric tests are run to provide a complete picture.

1. **T-test (parametric)**: Assumes normally distributed data. With n=15, the t-test is reasonably robust to mild non-normality.

2. **Mann-Whitney U test (non-parametric)**: Does not assume normal distributions. More conservative when normality cannot be verified.

**Test direction**: One-tailed tests were used to specifically test whether the memorized condition performs worse than other conditions:
- For similarity metrics (cosine, accuracy, agreement, probability): tests if memorized < other
- For KL divergence: tests if memorized > other (since higher KL divergence indicates greater distributional difference)

**Effect sizes**:
- Cohen's d (for t-test): d = (mean1 - mean2) / pooled_std
  - |d| < 0.2: small
  - |d| < 0.5: medium
  - |d| < 0.8: large
  - |d| ≥ 0.8: very large

- Rank-biserial correlation r (for Mann-Whitney): r = 1 - (2U)/(n1 × n2)
  - |r| < 0.1: negligible
  - |r| < 0.3: small
  - |r| < 0.5: medium
  - |r| ≥ 0.5: large

### Token Complexity Analysis

**Purpose**: To check whether predicted tokens differ in complexity across conditions, which could be a confounding variable.

**Metrics**:
- Zipf frequency: Word frequency on a logarithmic scale (higher = more common). Computed using the wordfreq library.
- Token length: Number of characters in the token.

**Tests used**: Same as metric significance analysis (t-test and Mann-Whitney U), but two-tailed since we're checking for any difference rather than a specific direction.

## Results

### Metric Means by Condition

| Metric | Memorized | Made-up | Random |
|--------|-----------|---------|--------|
| Last Token Cosine | 0.492 | 0.638 | 0.640 |
| Cumulative Cosine | 0.597 | 0.632 | 0.651 |
| Original Accuracy | 0.191 | 0.282 | 0.307 |
| Top-10 Agreement | 0.147 | 0.233 | 0.153 |
| Replacement P(Original Top) | 0.033 | 0.037 | 0.063 |
| KL Divergence | 10.73 | 6.89 | 6.56 |

### Token Prediction Match Rate

Percentage of prompts where the replacement model's top token matched the original model's top token:

| Condition | Match Rate |
|-----------|------------|
| Memorized | 6.7% (1/15) |
| Made-up | 0.0% (0/15) |
| Random | 6.7% (1/15) |

### Significance Analysis: Memorized vs Made-up

| Metric | t | t p-value | U | MW p-value | Cohen's d | Rank-biserial r |
|--------|---|-----------|---|------------|-----------|-----------------|
| Last Token Cosine | -1.93 | 0.032 | 75.5 | 0.065 | -0.73 (medium) | 0.33 (medium) |
| Cumulative Cosine | -1.32 | 0.099 | 81.0 | 0.099 | -0.50 (medium) | 0.28 (small) |
| Original Accuracy | -2.09 | 0.023 | 80.0 | 0.092 | -0.79 (medium) | 0.29 (small) |
| Top-10 Agreement | -1.13 | 0.134 | 101.5 | 0.327 | -0.43 (small) | 0.10 (negligible) |
| Replacement P(Original Top) | -0.11 | 0.458 | 95.0 | 0.240 | -0.04 (negligible) | 0.16 (small) |
| KL Divergence | 2.37 | 0.012 | 170.0 | 0.009 | 0.90 (large) | -0.51 (large) |

### Significance Analysis: Memorized vs Random

| Metric | t | t p-value | U | MW p-value | Cohen's d | Rank-biserial r |
|--------|---|-----------|---|------------|-----------|-----------------|
| Last Token Cosine | -1.84 | 0.038 | 75.0 | 0.062 | -0.70 (medium) | 0.33 (medium) |
| Cumulative Cosine | -1.86 | 0.037 | 64.0 | 0.023 | -0.70 (medium) | 0.43 (medium) |
| Original Accuracy | -2.72 | 0.006 | 58.5 | 0.013 | -1.03 (very large) | 0.48 (medium) |
| Top-10 Agreement | -0.13 | 0.451 | 109.0 | 0.447 | -0.05 (negligible) | 0.03 (negligible) |
| Replacement P(Original Top) | -0.49 | 0.315 | 82.0 | 0.107 | -0.18 (small) | 0.27 (small) |
| KL Divergence | 2.54 | 0.008 | 172.0 | 0.007 | 0.96 (large) | -0.53 (large) |

### Significance Analysis: Memorized vs Pooled (Made-up + Random)

This analysis compares memorized (n=15) against all other conditions combined (n=30), providing increased statistical power.

| Metric | t | t p-value | U | MW p-value | Cohen's d | Rank-biserial r |
|--------|---|-----------|---|------------|-----------|-----------------|
| Last Token Cosine | -2.35 | 0.012 | 150.5 | 0.037 | -0.76 (medium) | 0.33 (medium) |
| Cumulative Cosine | -1.87 | 0.034 | 145.0 | 0.028 | -0.61 (medium) | 0.36 (medium) |
| Original Accuracy | -2.68 | 0.005 | 138.5 | 0.019 | -0.87 (large) | 0.38 (medium) |
| Top-10 Agreement | -0.78 | 0.221 | 210.5 | 0.363 | -0.25 (small) | 0.06 (negligible) |
| Replacement P(Original Top) | -0.37 | 0.357 | 177.0 | 0.126 | -0.12 (negligible) | 0.21 (small) |
| KL Divergence | 3.12 | 0.002 | 342.0 | 0.003 | 1.01 (very large) | -0.52 (large) |

### Summary of Significant Results (p < 0.05)

#### Pairwise Comparisons

| Comparison | Metric | t-test | Mann-Whitney |
|------------|--------|--------|--------------|
| vs Made-up | Last Token Cosine | **Yes** | No |
| vs Made-up | Original Accuracy | **Yes** | No |
| vs Made-up | KL Divergence | **Yes** | **Yes** |
| vs Random | Last Token Cosine | **Yes** | No |
| vs Random | Cumulative Cosine | **Yes** | **Yes** |
| vs Random | Original Accuracy | **Yes** | **Yes** |
| vs Random | KL Divergence | **Yes** | **Yes** |

#### Pooled Comparison

| Comparison | Metric | t-test | Mann-Whitney |
|------------|--------|--------|--------------|
| vs Pooled | Last Token Cosine | **Yes** | **Yes** |
| vs Pooled | Cumulative Cosine | **Yes** | **Yes** |
| vs Pooled | Original Accuracy | **Yes** | **Yes** |
| vs Pooled | KL Divergence | **Yes** | **Yes** |

### Token Complexity Analysis

#### Means by Condition

| Metric | Memorized | Made-up | Random |
|--------|-----------|---------|--------|
| Zipf Frequency | 4.87 ± 0.88 | 5.19 ± 0.94 | 4.88 ± 0.87 |
| Token Length | 4.93 ± 2.08 | 4.33 ± 1.58 | 5.20 ± 1.60 |

#### Memorized vs Made-up

| Metric | t | t p-value | U | MW p-value | Cohen's d | Significant |
|--------|---|-----------|---|------------|-----------|-------------|
| Zipf Frequency | -0.93 | 0.361 | 92.5 | 0.418 | -0.35 (small) | No |
| Token Length | 0.86 | 0.397 | 123.0 | 0.667 | 0.32 (small) | No |

#### Memorized vs Random

| Metric | t | t p-value | U | MW p-value | Cohen's d | Significant |
|--------|---|-----------|---|------------|-----------|-------------|
| Zipf Frequency | -0.02 | 0.982 | 114.5 | 0.950 | -0.01 (negligible) | No |
| Token Length | -0.38 | 0.707 | 92.5 | 0.407 | -0.14 (negligible) | No |

No significant differences in token complexity between memorized and other conditions.

## Conclusion

The memorized condition shows lower fidelity between the base model and replacement model compared to the made-up and random conditions across multiple metrics.

**Significant findings:**

- **KL Divergence**: The memorized condition has significantly higher KL divergence (10.73) compared to both made-up (6.89) and random (6.56). This result is significant on both the t-test and Mann-Whitney U test, with large effect sizes (Cohen's d ≈ 0.9, rank-biserial r ≈ 0.5).

- **Original Accuracy**: The memorized condition has significantly lower argmax prediction match rate (0.191) compared to random (0.307). Significant on both tests with a very large effect size (Cohen's d = -1.03).

- **Cumulative Cosine**: The memorized condition has significantly lower average cosine similarity (0.597) compared to random (0.651). Significant on both tests with medium effect sizes.

- **Last Token Cosine**: The memorized condition shows lower cosine similarity at the final position. Significant on the t-test for both comparisons, though not on Mann-Whitney U.

**Pooled analysis:**

The pooled comparison (memorized vs all other conditions combined) shows significance on both tests for four metrics: Last Token Cosine, Cumulative Cosine, Original Accuracy, and KL Divergence. The increased sample size (n=30 for pooled) provides greater statistical power, resulting in both t-test and Mann-Whitney U reaching significance for metrics that were only t-test significant in pairwise comparisons.

**Non-significant metrics:**

- Top-10 Agreement and Replacement P(Original Top) do not show significant differences between conditions in either pairwise or pooled analyses.

**Confound check:**

Token complexity (Zipf frequency and token length) does not differ significantly between conditions, indicating that differences in token difficulty are not driving the observed effects.

## Output Files

- `error_hypothesis_analysis.json` - Raw metrics for all conditions and configs
- `significance_analysis.csv` - Pairwise statistical test results
- `pooled_significance_analysis.csv` - Pooled statistical test results (memorized vs all others)
- `token_predictions.csv` - Token prediction comparisons
- `token_complexity/token_complexity.csv` - Token complexity data
- `token_complexity/complexity_significance.csv` - Complexity statistical tests
- `bar_charts/` - Bar charts for each metric
- `boxplots/` - Boxplots for each metric
- `heatmaps/` - Heatmaps showing metrics by config and condition
- `token_complexity/` - Token complexity visualizations
- `significance_viz/` - Effect size visualizations for significance analysis
  - `pooled_effect_sizes.png` - Memorized vs pooled (made-up + random) effect sizes
  - `memorized_vs_made_up_effect_sizes.png` - Memorized vs made-up effect sizes
  - `memorized_vs_random_effect_sizes.png` - Memorized vs random effect sizes
