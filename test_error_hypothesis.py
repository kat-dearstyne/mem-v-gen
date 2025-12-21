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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformer_lens as tl
from dotenv import load_dotenv
from huggingface_hub import login

from circuit_tracer.replacement_model import ReplacementModel


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

IS_TEST = os.getenv("IS_TEST", "false").lower() == "true"
ANALYZE_RESULTS = os.getenv("ANALYZE_RESULTS", "false").lower() == "true"

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

    # Metrics bounded 0-1
    bounded_metrics = ["last_token_cosine", "cumulative_cosine", "original_accuracy",
                       "top_k_agreement", "replacement_prob_of_original_top"]
    bounded_titles = ["Last Token Cosine", "Cumulative Cosine", "Original Accuracy",
                      f"Top-{TOP_K} Agreement", "Replacement P(Original Top)"]

    # 1. Bar chart: Average metrics by condition (bounded metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(bounded_metrics, bounded_titles)):
        ax = axes[i]
        means = df.groupby("condition")[metric].mean().reindex(conditions)
        stds = df.groupby("condition")[metric].std().reindex(conditions)

        ax.bar(range(len(conditions)), means.values, color=palette, alpha=0.8)
        ax.errorbar(range(len(conditions)), means.values, yerr=stds.values,
                    fmt='none', color='black', capsize=5)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)

    # KL divergence (unbounded, separate scale)
    ax = axes[5]
    means = df.groupby("condition")["kl_divergence"].mean().reindex(conditions)
    stds = df.groupby("condition")["kl_divergence"].std().reindex(conditions)
    ax.bar(range(len(conditions)), means.values, color=palette, alpha=0.8)
    ax.errorbar(range(len(conditions)), means.values, yerr=stds.values,
                fmt='none', color='black', capsize=5)
    ax.set_title("KL Divergence", fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel("KL Divergence (nats)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_condition.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Boxplot: Distribution of metrics by condition
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(bounded_metrics, bounded_titles)):
        ax = axes[i]
        sns.boxplot(data=df, x="condition", y=metric, order=conditions,
                    palette=palette, ax=ax)
        sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                      color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel("Score")
        ax.set_xticklabels(conditions, rotation=45, ha='right')

    ax = axes[5]
    sns.boxplot(data=df, x="condition", y="kl_divergence", order=conditions,
                palette=palette, ax=ax)
    sns.stripplot(data=df, x="condition", y="kl_divergence", order=conditions,
                  color='black', alpha=0.5, size=4, ax=ax)
    ax.set_title("KL Divergence", fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel("KL Divergence (nats)")
    ax.set_xticklabels(conditions, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Heatmap: Metrics by config and condition
    all_metrics = bounded_metrics + ["kl_divergence"]
    all_titles = bounded_titles + ["KL Divergence"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(all_metrics, all_titles)):
        ax = axes[i]
        pivot = df.pivot(index="config", columns="condition", values=metric)
        pivot = pivot[conditions]  # Reorder columns
        vmin, vmax = (0, 1) if metric != "kl_divergence" else (None, None)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label': 'Score'})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel("Config")

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Token prediction comparison table
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

    # Save token comparison as CSV
    token_df.to_csv(output_dir / "token_predictions.csv", index=False)

    # Summary: match rate by condition
    match_rate = token_df.groupby("condition")["match"].mean()
    print("\nToken Prediction Match Rate by Condition:")
    for condition in conditions:
        if condition in match_rate.index:
            print(f"  {condition}: {match_rate[condition]:.2%}")

    print(f"\nVisualizations saved to: {output_dir}")


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
    if ANALYZE_RESULTS:
        analyze_results()
    else:
        main()
