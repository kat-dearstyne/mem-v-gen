"""
Test error hypothesis: Compare base model logits with CLT replacement logits.

Measures accuracy between original model outputs and outputs when using
transcoder-reconstructed MLP activations.
"""

import json
import os
from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformer_lens as tl
from dotenv import load_dotenv
from huggingface_hub import login
from scipy import stats
from wordfreq import zipf_frequency

from circuit_tracer.replacement_model import ReplacementModel
from utils import get_env_bool
from visualizations import (
    plot_error_hypothesis_metrics,
    plot_error_hypothesis_combined_boxplot,
    plot_token_complexity,
    plot_significance_effect_sizes,
)


# Condition names mapping to config structure:
# CONDITIONS[0] -> MAIN_PROMPT
# CONDITIONS[1] -> DIFF_PROMPTS[0]
# CONDITIONS[2] -> DIFF_PROMPTS[1]
# etc.
# Set a condition to None to skip it
CONDITIONS = ["memorized", None, "made-up", "random"]

CONFIG_DIR = Path("configs")

OUTPUT_DIR = Path("output/test_error_hypothesis")
load_dotenv()
# Metrics for a single prompt's accuracy test
AccuracyMetrics = namedtuple("AccuracyMetrics", [
    "last_token_cosine",
    "cumulative_cosine",
    "original_accuracy",
    "original_top_token",
    "replacement_top_token",
    "kl_divergence",
    "top_k_agreement",
    "replacement_prob_of_original_top",
])

TOP_K = 10  # For top-k agreement metric

IS_TEST = get_env_bool("IS_TEST", False)
ANALYZE_RESULTS = get_env_bool("ANALYZE_RESULTS", False)
RUN_SANITY_CHECK = get_env_bool("RUN_SANITY_CHECK", False)

def get_replacement_logits(model, prompt_tokens):
    """
    Get logits using transcoder-reconstructed MLP activations.

    Hooks into each layer to encode activations through transcoders,
    then decode them back, replacing the original MLP outputs.
    """
    features = torch.zeros(
        (model.cfg.n_layers, len(prompt_tokens), model.transcoders.d_transcoder),
        dtype=torch.bfloat16
    ).to(model.cfg.device)
    bos_actv = torch.zeros((model.cfg.d_model), dtype=torch.bfloat16).to(model.cfg.device)

    def input_hook_fn(value, hook, layer):
        nonlocal bos_actv
        bos_actv = value[0, 0]
        features[layer] = model.transcoders.encode_layer(value, layer)
        features[:, 0] = 0.  # exclude bos
        return value

    def output_hook_fn(value, hook, layer):
        mlp_outs = model.transcoders.decode(features)
        mlp_out = mlp_outs[layer].unsqueeze(0)
        mlp_out[:, 0] = bos_actv
        return mlp_out

    all_hooks = []
    for layer in range(model.cfg.n_layers):
        input_hook_fn_partial = partial(input_hook_fn, layer=layer)
        all_hooks.append((tl.utils.get_act_name(model.feature_input_hook[5:], layer), input_hook_fn_partial))

        output_hook_fn_partial = partial(output_hook_fn, layer=layer)
        all_hooks.append((tl.utils.get_act_name(model.feature_output_hook[5:], layer), output_hook_fn_partial))

    with torch.no_grad():
        logits = model.run_with_hooks(prompt_tokens, fwd_hooks=all_hooks, return_type='logits')
    return logits


def get_last_token_accuracy(base_logits_BPV, replacement_logits_BPV):
    """
    Compute accuracy as cosine similarity between logit vectors at last position.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)

    Returns:
        Cosine similarity between final position logits.
    """
    base_logits_V = base_logits_BPV[0, -1]
    replacement_logits_V = replacement_logits_BPV[0, -1]

    base_norm = torch.linalg.norm(base_logits_V)
    replacement_norm = torch.linalg.norm(replacement_logits_V)

    accuracy = (base_logits_V.T @ replacement_logits_V.T) / (base_norm * replacement_norm)
    accuracy = torch.abs(accuracy).item()

    return accuracy


def get_cumulative_token_accuracy(base_logits_BPV, replacement_logits_BPV):
    """
    Compute average cosine similarity across all token positions.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)

    Returns:
        Average cosine similarity across all positions.
    """
    accuracy = 0
    num_tokens = base_logits_BPV.shape[1]

    for i in range(num_tokens):
        base_logits_V = base_logits_BPV[0, i]
        replacement_logits_V = replacement_logits_BPV[0, i]

        base_norm = torch.linalg.norm(base_logits_V)
        replacement_norm = torch.linalg.norm(replacement_logits_V)

        cosine_distance = (base_logits_V.T @ replacement_logits_V.T) / (base_norm * replacement_norm)
        cosine_distance = torch.abs(cosine_distance).item()

        accuracy += cosine_distance

    accuracy /= num_tokens
    return accuracy


def get_original_accuracy_metric(base_logits_BPV, replacement_logits_BPV, prompt_tokens):
    """
    Compute accuracy by directly comparing argmax predictions.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)
        prompt_tokens: Tokenized prompt

    Returns:
        Fraction of positions where argmax predictions match.
    """
    repl_acc = (base_logits_BPV.argmax(dim=-1) == replacement_logits_BPV.argmax(dim=-1)).sum() / prompt_tokens.numel()
    return repl_acc.item()


def get_kl_divergence(base_logits_BPV, replacement_logits_BPV):
    """
    Compute KL divergence between probability distributions at the last position.

    KL(base || replacement) measures how much information is lost when using
    the replacement distribution to approximate the base distribution.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)

    Returns:
        KL divergence (in nats) at the final position.
    """
    base_logits = base_logits_BPV[0, -1].float()
    replacement_logits = replacement_logits_BPV[0, -1].float()

    base_probs = torch.softmax(base_logits, dim=-1)
    replacement_log_probs = torch.log_softmax(replacement_logits, dim=-1)

    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
    kl_div = torch.sum(base_probs * (torch.log(base_probs + 1e-10) - replacement_log_probs))
    return kl_div.item()


def get_top_k_agreement(base_logits_BPV, replacement_logits_BPV, k: int = TOP_K):
    """
    Compute the fraction of top-k tokens that overlap between distributions.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)
        k: Number of top tokens to compare

    Returns:
        Fraction of top-k tokens that appear in both distributions (0 to 1).
    """
    base_logits = base_logits_BPV[0, -1]
    replacement_logits = replacement_logits_BPV[0, -1]

    base_top_k = torch.topk(base_logits, k).indices.tolist()
    replacement_top_k = torch.topk(replacement_logits, k).indices.tolist()

    overlap = len(set(base_top_k) & set(replacement_top_k))
    return overlap / k


def get_replacement_prob_of_original_top(base_logits_BPV, replacement_logits_BPV):
    """
    Get the probability the replacement model assigns to the original model's top prediction.

    Args:
        base_logits_BPV: Base model logits (batch, position, vocab)
        replacement_logits_BPV: Replacement model logits (batch, position, vocab)

    Returns:
        Probability (0 to 1) that replacement model assigns to original's top token.
    """
    base_logits = base_logits_BPV[0, -1]
    replacement_logits = replacement_logits_BPV[0, -1].float()

    original_top_token_id = base_logits.argmax().item()
    replacement_probs = torch.softmax(replacement_logits, dim=-1)

    return replacement_probs[original_top_token_id].item()


def load_model():
    """Load the ReplacementModel with CLT transcoders."""
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
    if IS_TEST:
        return None

    model = ReplacementModel.from_pretrained(
        "google/gemma-2-2b",
        "mntss/clt-gemma-2-2b-426k",
        dtype=torch.bfloat16,
    )
    return model


def run_accuracy_test(model, prompt_str: str) -> AccuracyMetrics:
    """
    Run accuracy comparison for a single prompt.

    Args:
        model: The ReplacementModel instance
        prompt_str: The prompt string to test

    Returns:
        AccuracyMetrics namedtuple with all accuracy measurements
    """
    if IS_TEST:
        # Dummy tensors for testing without model
        prompt_tokens = torch.zeros((1, 10), dtype=torch.long)
        base_logits_BPV = torch.randn((1, 10, 256000), dtype=torch.bfloat16)
        replacement_logits_BPV = torch.randn((1, 10, 256000), dtype=torch.bfloat16)
        original_top_token = "<test>"
        replacement_top_token = "<test>"
    else:
        prompt_tokens = model.ensure_tokenized(prompt_str)

        with torch.no_grad():
            base_logits_BPV = model(prompt_tokens, return_type='logits')
            replacement_logits_BPV = get_replacement_logits(model, prompt_tokens)

        # Get top token predictions at final position for qualitative comparison
        original_top_token_id = base_logits_BPV[0, -1].argmax().item()
        replacement_top_token_id = replacement_logits_BPV[0, -1].argmax().item()
        original_top_token = model.tokenizer.decode([original_top_token_id])
        replacement_top_token = model.tokenizer.decode([replacement_top_token_id])

    last_token_acc = get_last_token_accuracy(base_logits_BPV, replacement_logits_BPV)
    cumulative_acc = get_cumulative_token_accuracy(base_logits_BPV, replacement_logits_BPV)
    orig_acc = get_original_accuracy_metric(base_logits_BPV, replacement_logits_BPV, prompt_tokens)
    kl_div = get_kl_divergence(base_logits_BPV, replacement_logits_BPV)
    top_k_agree = get_top_k_agreement(base_logits_BPV, replacement_logits_BPV)
    repl_prob_orig = get_replacement_prob_of_original_top(base_logits_BPV, replacement_logits_BPV)

    return AccuracyMetrics(
        last_token_cosine=last_token_acc,
        cumulative_cosine=cumulative_acc,
        original_accuracy=orig_acc,
        original_top_token=original_top_token,
        replacement_top_token=replacement_top_token,
        kl_divergence=kl_div,
        top_k_agreement=top_k_agree,
        replacement_prob_of_original_top=repl_prob_orig,
    )


def load_config(config_path: Path) -> dict:
    """Load a config file and return the parsed JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_prompt_for_condition(config: dict, condition_index: int) -> Optional[str]:
    """
    Get the prompt string for a given condition index.

    Args:
        config: Parsed config dictionary
        condition_index: Index into CONDITIONS list

    Returns:
        The prompt string, or None if not available
    """
    if condition_index == 0:
        return config.get("MAIN_PROMPT")
    else:
        diff_prompts = config.get("DIFF_PROMPTS", [])
        diff_index = condition_index - 1
        if diff_index < len(diff_prompts):
            return diff_prompts[diff_index]
        return None


def run_analysis_for_configs(model, config_dir: Path = CONFIG_DIR) -> dict:
    """
    Run accuracy analysis across all configs and conditions.

    Args:
        model: The ReplacementModel instance
        config_dir: Directory containing config JSON files

    Returns:
        Dictionary structured as {condition: {filename: AccuracyMetrics}}
    """
    results = {cond: {} for cond in CONDITIONS if cond is not None}

    config_files = sorted(config_dir.glob("*.json"))

    for config_path in config_files:
        config_name = config_path.stem
        config = load_config(config_path)

        print(f"\nProcessing config: {config_name}")

        for cond_idx, condition in enumerate(CONDITIONS):
            if condition is None:
                continue

            prompt = get_prompt_for_condition(config, cond_idx)
            if prompt is None:
                print(f"  {condition}: No prompt available, skipping")
                continue

            print(f"  {condition}: {prompt[:50]}...")
            metrics = run_accuracy_test(model, prompt)
            results[condition][config_name] = metrics

    return results


def save_results(results: dict, output_dir: Path = OUTPUT_DIR):
    """
    Save results to JSON file.

    Args:
        results: Dictionary structured as {condition: {filename: AccuracyMetrics}}
        output_dir: Directory to save the output file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "error_hypothesis_analysis.json"

    # Convert namedtuples to dicts for JSON serialization
    serializable_results = {}
    for condition, config_metrics in results.items():
        serializable_results[condition] = {
            config_name: metrics._asdict()
            for config_name, metrics in config_metrics.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def load_results(output_dir: Path = OUTPUT_DIR) -> dict:
    """
    Load results from JSON file.

    Args:
        output_dir: Directory containing the results file

    Returns:
        Dictionary structured as {condition: {filename: metrics_dict}}
    """
    output_path = output_dir / "error_hypothesis_analysis.json"
    with open(output_path, "r") as f:
        return json.load(f)


def get_token_complexity(token: str) -> dict:
    """
    Get complexity metrics for a token.

    Args:
        token: The token string (may include leading space from tokenizer)

    Returns:
        Dictionary with zipf_frequency and token_length
    """
    # Clean token (remove leading/trailing whitespace for frequency lookup)
    clean_token = token.strip()

    # Zipf frequency (scale ~0-8, higher = more common)
    # Returns 0 for unknown words
    freq = zipf_frequency(clean_token, 'en') if clean_token else 0.0

    return {
        "zipf_frequency": freq,
        "token_length": len(clean_token),
    }


HIGHER_IS_BETTER_METRICS = ["last_token_cosine", "cumulative_cosine", "original_accuracy",
                            "top_k_agreement", "replacement_prob_of_original_top"]
LOWER_IS_BETTER_METRICS = ["kl_divergence"]
ALL_METRICS = HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS


def compute_pooled_significance(
    df: pd.DataFrame,
    target_condition: str,
    other_conditions: list,
    test_target_worse: bool = True,
) -> pd.DataFrame:
    """
    Compute significance comparing target condition against ALL other conditions pooled.

    Args:
        df: DataFrame with columns: condition and metric columns
        target_condition: The condition to compare against pooled others
        other_conditions: List of conditions to pool together
        test_target_worse: If True, test if target performs worse (lower similarity, higher KL).
                          If False, test if target performs better.

    Returns:
        DataFrame with significance results (one row per metric)
    """
    available_conditions = df["condition"].unique().tolist()
    if target_condition not in available_conditions:
        return pd.DataFrame()

    other_conditions = [c for c in other_conditions if c in available_conditions]
    if not other_conditions:
        return pd.DataFrame()

    target_data = df[df["condition"] == target_condition]
    pooled_data = df[df["condition"].isin(other_conditions)]

    results_rows = []

    for metric in ALL_METRICS:
        target_values = target_data[metric].values
        pooled_values = pooled_data[metric].values
        is_higher_better = metric in HIGHER_IS_BETTER_METRICS

        # Determine test direction
        if test_target_worse:
            alternative = 'less' if is_higher_better else 'greater'
        else:
            alternative = 'greater' if is_higher_better else 'less'

        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p = stats.mannwhitneyu(target_values, pooled_values, alternative=alternative)

        # T-test (parametric)
        t_stat, t_p = stats.ttest_ind(target_values, pooled_values, alternative=alternative)

        # Effect size: Cohen's d
        pooled_std = np.sqrt(((len(target_values)-1)*target_values.std()**2 +
                              (len(pooled_values)-1)*pooled_values.std()**2) /
                             (len(target_values) + len(pooled_values) - 2))
        cohens_d = (target_values.mean() - pooled_values.mean()) / pooled_std if pooled_std > 0 else 0

        # Effect size: rank-biserial correlation (for Mann-Whitney)
        n1, n2 = len(target_values), len(pooled_values)
        rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)

        results_rows.append({
            "metric": metric,
            "comparison": f"{target_condition} vs pooled({'+'.join(other_conditions)})",
            "target_condition": target_condition,
            "other_conditions": "+".join(other_conditions),
            "test_direction": "target_worse" if test_target_worse else "target_better",
            "target_mean": target_values.mean(),
            "target_std": target_values.std(),
            "pooled_mean": pooled_values.mean(),
            "pooled_std": pooled_values.std(),
            "n_target": n1,
            "n_pooled": n2,
            "t_statistic": t_stat,
            "t_p_value": t_p,
            "t_significant": t_p < 0.05,
            "mann_whitney_u": mw_stat,
            "mw_p_value": mw_p,
            "mw_significant": mw_p < 0.05,
            "cohens_d": cohens_d,
            "rank_biserial_r": rank_biserial,
        })

    return pd.DataFrame(results_rows)


def compute_pairwise_significance(
    df: pd.DataFrame,
    target_condition: str,
    other_conditions: list,
    test_target_worse: bool = True,
) -> pd.DataFrame:
    """
    Core function to compute pairwise significance between conditions.

    Args:
        df: DataFrame with columns: condition and metric columns
        target_condition: The condition to compare against others
        other_conditions: List of conditions to compare target against
        test_target_worse: If True, test if target performs worse (lower similarity, higher KL).
                          If False, test if target performs better.

    Returns:
        DataFrame with significance results
    """
    available_conditions = df["condition"].unique().tolist()
    if target_condition not in available_conditions:
        return pd.DataFrame()

    other_conditions = [c for c in other_conditions if c in available_conditions]
    if not other_conditions:
        return pd.DataFrame()

    target_data = df[df["condition"] == target_condition]
    results_rows = []

    for metric in ALL_METRICS:
        target_values = target_data[metric].values
        is_higher_better = metric in HIGHER_IS_BETTER_METRICS

        for other_cond in other_conditions:
            other_values = df[df["condition"] == other_cond][metric].values

            # Determine test direction
            # test_target_worse=True: for higher_is_better metrics, test target < other (alternative='less')
            # test_target_worse=False: for higher_is_better metrics, test target > other (alternative='greater')
            if test_target_worse:
                alternative = 'less' if is_higher_better else 'greater'
            else:
                alternative = 'greater' if is_higher_better else 'less'

            # Mann-Whitney U test (non-parametric)
            mw_stat, mw_p = stats.mannwhitneyu(target_values, other_values, alternative=alternative)

            # T-test (parametric)
            t_stat, t_p = stats.ttest_ind(target_values, other_values, alternative=alternative)

            # Effect size: Cohen's d
            pooled_std = np.sqrt(((len(target_values)-1)*target_values.std()**2 +
                                  (len(other_values)-1)*other_values.std()**2) /
                                 (len(target_values) + len(other_values) - 2))
            cohens_d = (target_values.mean() - other_values.mean()) / pooled_std if pooled_std > 0 else 0

            # Effect size: rank-biserial correlation (for Mann-Whitney)
            n1, n2 = len(target_values), len(other_values)
            rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)

            results_rows.append({
                "metric": metric,
                "comparison": f"{target_condition} vs {other_cond}",
                "target_condition": target_condition,
                "other_condition": other_cond,
                "test_direction": "target_worse" if test_target_worse else "target_better",
                "target_mean": target_values.mean(),
                "target_std": target_values.std(),
                "other_mean": other_values.mean(),
                "other_std": other_values.std(),
                "n_per_group": n1,
                "t_statistic": t_stat,
                "t_p_value": t_p,
                "t_significant": t_p < 0.05,
                "mann_whitney_u": mw_stat,
                "mw_p_value": mw_p,
                "mw_significant": mw_p < 0.05,
                "cohens_d": cohens_d,
                "rank_biserial_r": rank_biserial,
            })

    return pd.DataFrame(results_rows)


def print_significant_results(results_df: pd.DataFrame, label: str):
    """
    Print significant results from a significance analysis DataFrame.

    Args:
        results_df: DataFrame with t_significant, mw_significant, metric, comparison columns
        label: Label to identify the type of analysis (e.g., "Pairwise", "Pooled")
    """
    if results_df.empty:
        print(f"{label}: No results")
        return

    t_significant = results_df[results_df["t_significant"] == True][["metric", "comparison"]]
    mw_significant = results_df[results_df["mw_significant"] == True][["metric", "comparison"]]
    print(f"{label} T-Test Significant Results: ", t_significant)
    print(f"{label} Mann Whitney Significant Results: ", mw_significant)


def analyze_metric_significance(df: pd.DataFrame, output_dir: Path) -> tuple:
    """
    Analyze statistical significance of metrics comparing memorized vs other conditions.

    Tests whether memorized condition performs worse than other conditions.
    Saves pairwise results to significance_analysis.csv and pooled results to
    pooled_significance_analysis.csv.

    Args:
        df: DataFrame with columns: condition and metric columns
        output_dir: Directory to save results CSV

    Returns:
        Tuple of (pairwise_results_df, pooled_results_df)
    """
    conditions = df["condition"].unique().tolist()
    other_conditions = [c for c in conditions if c != "memorized"]

    # Pairwise comparisons (memorized vs each other condition separately)
    pairwise_df = compute_pairwise_significance(
        df,
        target_condition="memorized",
        other_conditions=other_conditions,
        test_target_worse=True,
    )

    if not pairwise_df.empty:
        pairwise_df.to_csv(output_dir / "significance_analysis.csv", index=False)

    # Pooled comparison (memorized vs all other conditions combined)
    pooled_df = compute_pooled_significance(
        df,
        target_condition="memorized",
        other_conditions=other_conditions,
        test_target_worse=True,
    )

    if not pooled_df.empty:
        pooled_df.to_csv(output_dir / "pooled_significance_analysis.csv", index=False)

    return pairwise_df, pooled_df


def sanity_check_significance(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Run sanity check significance tests to verify results.

    Tests:
    1. Memorized vs others with FLIPPED direction (testing if memorized is better)
    2. Made-up vs random (testing if non-memorized conditions differ)

    These should generally NOT be significant if the main results are valid.

    Args:
        df: DataFrame with columns: condition and metric columns
        output_dir: Directory to save results CSV
    """
    conditions = df["condition"].unique().tolist()
    all_results = []

    # Test 1: Memorized performing BETTER (should NOT be significant)
    other_conditions = [c for c in conditions if c != "memorized"]
    flipped_results = compute_pairwise_significance(
        df,
        target_condition="memorized",
        other_conditions=other_conditions,
        test_target_worse=False,  # Flipped direction
    )
    if not flipped_results.empty:
        flipped_results["sanity_check"] = "memorized_better (flipped)"
        all_results.append(flipped_results)

    # Test 2: Made-up vs random (should NOT be significant)
    if "made-up" in conditions and "random" in conditions:
        madeup_vs_random = compute_pairwise_significance(
            df,
            target_condition="made-up",
            other_conditions=["random"],
            test_target_worse=True,
        )
        if not madeup_vs_random.empty:
            madeup_vs_random["sanity_check"] = "made-up_worse_than_random"
            all_results.append(madeup_vs_random)

        random_vs_madeup = compute_pairwise_significance(
            df,
            target_condition="random",
            other_conditions=["made-up"],
            test_target_worse=True,
        )
        if not random_vs_madeup.empty:
            random_vs_madeup["sanity_check"] = "random_worse_than_made-up"
            all_results.append(random_vs_madeup)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        sanity_dir = output_dir / "sanity_checks"
        sanity_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(sanity_dir / "sanity_check_significance.csv", index=False)
        return results_df

    return pd.DataFrame()


def analyze_token_complexity(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze token complexity across conditions to check for confounds.

    Adds complexity metrics to df, creates visualizations, runs statistical tests,
    and saves results to CSV.

    Args:
        df: DataFrame with columns: condition, config, original_top_token
        output_dir: Directory to save visualizations and CSV
    """
    # Add token complexity metrics
    df["zipf_frequency"] = df["original_top_token"].apply(
        lambda t: get_token_complexity(t)["zipf_frequency"]
    )
    df["token_length"] = df["original_top_token"].apply(
        lambda t: get_token_complexity(t)["token_length"]
    )

    conditions = df["condition"].unique().tolist()
    palette = sns.color_palette("husl", len(conditions))

    # Create visualizations
    plot_token_complexity(df, output_dir, conditions, palette)

    # Statistical tests - pairwise comparisons like metric significance
    complexity_metrics = ["zipf_frequency", "token_length"]
    results_rows = []

    if "memorized" not in conditions:
        return pd.DataFrame()

    other_conditions = [c for c in conditions if c != "memorized"]
    if not other_conditions:
        return pd.DataFrame()

    memorized_data = df[df["condition"] == "memorized"]

    for metric in complexity_metrics:
        mem_values = memorized_data[metric].values

        for other_cond in other_conditions:
            other_values = df[df["condition"] == other_cond][metric].values

            # T-test (two-tailed - just checking for difference, not direction)
            t_stat, t_p = stats.ttest_ind(mem_values, other_values)

            # Mann-Whitney U test (two-tailed)
            mw_stat, mw_p = stats.mannwhitneyu(mem_values, other_values, alternative='two-sided')

            # Effect size: Cohen's d
            pooled_std = np.sqrt(((len(mem_values)-1)*mem_values.std()**2 +
                                  (len(other_values)-1)*other_values.std()**2) /
                                 (len(mem_values) + len(other_values) - 2))
            cohens_d = (mem_values.mean() - other_values.mean()) / pooled_std if pooled_std > 0 else 0

            # Effect size: rank-biserial correlation
            n1, n2 = len(mem_values), len(other_values)
            rank_biserial = 1 - (2 * mw_stat) / (n1 * n2)

            results_rows.append({
                "metric": metric,
                "comparison": f"memorized vs {other_cond}",
                "memorized_mean": mem_values.mean(),
                "memorized_std": mem_values.std(),
                "other_mean": other_values.mean(),
                "other_std": other_values.std(),
                "n_per_group": n1,
                "t_statistic": t_stat,
                "t_p_value": t_p,
                "t_significant": t_p < 0.05,
                "mann_whitney_u": mw_stat,
                "mw_p_value": mw_p,
                "mw_significant": mw_p < 0.05,
                "cohens_d": cohens_d,
                "rank_biserial_r": rank_biserial,
            })

    # Save complexity data
    complexity_dir = output_dir / "token_complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)

    complexity_df = df[["condition", "config", "original_top_token", "zipf_frequency", "token_length"]]
    complexity_df.to_csv(complexity_dir / "token_complexity.csv", index=False)

    results_df = pd.DataFrame(results_rows)
    if not results_df.empty:
        results_df.to_csv(complexity_dir / "complexity_significance.csv", index=False)

    return results_df


def analyze_results(output_dir: Path = OUTPUT_DIR):
    """
    Load results and create visualizations comparing conditions.

    Args:
        output_dir: Directory containing results and where to save visualizations
    """
    results = load_results(output_dir)

    # Convert to DataFrame for easier plotting
    rows = []
    for condition, config_metrics in results.items():
        for config_name, metrics in config_metrics.items():
            rows.append({
                "condition": condition,
                "config": config_name,
                **metrics
            })
    df = pd.DataFrame(rows)

    if df.empty:
        print("No results to analyze")
        return

    conditions = df["condition"].unique().tolist()
    palette = sns.color_palette("husl", len(conditions))

    # Create metric visualizations (bar charts, boxplots, heatmaps)
    plot_error_hypothesis_metrics(df, output_dir, top_k=TOP_K)

    # Create combined boxplot for main metrics
    plot_error_hypothesis_combined_boxplot(
        df,
        conditions=conditions,
        palette=palette,
        save_path=output_dir / "combined_metrics_boxplot.png",
    )

    # Token prediction comparison table
    token_rows = []
    for condition, config_metrics in results.items():
        for config_name, metrics in config_metrics.items():
            token_rows.append({
                "condition": condition,
                "config": config_name,
                "original": metrics["original_top_token"],
                "replacement": metrics["replacement_top_token"],
                "match": metrics["original_top_token"] == metrics["replacement_top_token"]
            })
    token_df = pd.DataFrame(token_rows)
    token_df.to_csv(output_dir / "token_predictions.csv", index=False)

    # Statistical significance analysis
    pairwise_df, pooled_df = analyze_metric_significance(df, output_dir)
    print_significant_results(pairwise_df, "Pairwise")
    print_significant_results(pooled_df, "Pooled")

    # Significance effect size visualizations
    exclude_metrics = ["top_k_agreement", "replacement_prob_of_original_top"]
    sig_viz_dir = output_dir / "significance_viz"
    sig_viz_dir.mkdir(parents=True, exist_ok=True)

    # Pooled visualization
    if not pooled_df.empty:
        plot_significance_effect_sizes(
            pooled_df,
            title="Memorized vs Pooled (Made-up + Random)",
            save_path=sig_viz_dir / "pooled_effect_sizes.png",
            exclude_metrics=exclude_metrics,
        )

    # Pairwise visualizations (separate file for each comparison)
    if not pairwise_df.empty:
        for comparison in pairwise_df["comparison"].unique():
            comparison_df = pairwise_df[pairwise_df["comparison"] == comparison]
            # Create filename-safe version of comparison name
            safe_name = comparison.replace(" ", "_").replace("-", "_")
            plot_significance_effect_sizes(
                comparison_df,
                title=comparison.replace("_", " ").title(),
                save_path=sig_viz_dir / f"{safe_name}_effect_sizes.png",
                exclude_metrics=exclude_metrics,
            )

    # Token complexity analysis
    analyze_token_complexity(df, output_dir)

    print(f"\nResults saved to: {output_dir}")


def run_sanity_checks(output_dir: Path = OUTPUT_DIR):
    """
    Run sanity check significance tests on existing results.

    Loads results and runs tests that should NOT be significant:
    1. Memorized performing better than other conditions (flipped direction)
    2. Made-up vs random comparisons

    Args:
        output_dir: Directory containing results
    """
    results = load_results(output_dir)

    # Convert to DataFrame
    rows = []
    for condition, config_metrics in results.items():
        for config_name, metrics in config_metrics.items():
            rows.append({
                "condition": condition,
                "config": config_name,
                **metrics
            })
    df = pd.DataFrame(rows)

    if df.empty:
        print("No results to analyze")
        return

    sanity_df = sanity_check_significance(df, output_dir)

    if sanity_df.empty:
        print("No sanity check results generated")
        return

    # Report any unexpected significant results
    sanity_sig = sanity_df[
        (sanity_df["t_significant"]) | (sanity_df["mw_significant"])
    ][["sanity_check", "metric", "comparison", "t_p_value", "mw_p_value"]]

    if not sanity_sig.empty:
        print("\nSanity Check - Unexpected Significant Results:")
        print(sanity_sig.to_string(index=False))
    else:
        print("\nSanity Check: No unexpected significant results (good)")

    print(f"\nSanity check results saved to: {output_dir / 'sanity_checks'}")


def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded.\n")

    print(f"Conditions: {[c for c in CONDITIONS if c is not None]}")
    print(f"Config directory: {CONFIG_DIR}")

    results = run_analysis_for_configs(model, CONFIG_DIR)
    save_results(results)

    return results


if __name__ == "__main__":
    if RUN_SANITY_CHECK:
        run_sanity_checks()
    elif ANALYZE_RESULTS:
        analyze_results()
    else:
        main()
