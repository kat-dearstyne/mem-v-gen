import random
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, ttest_rel

from constants import COLORS, CUSTOM_PALETTE

# Set up plot styling
sns.set_theme(style="whitegrid")
sns.set_palette(CUSTOM_PALETTE)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
})

from attribution_graph_utils import create_or_load_graph
from common_utils import get_id_without_pos, get_top_output_logit_node, get_output_logits, get_links_from_node, \
    get_nodes_linked_to_target, get_node_dict

CONDITION1 = "memorized"
CONDITION2 = "non-memorized"


def run_error_ranking(prompt1: str, prompt2: str, model: str, submodel: str,
                      graph_dir: Optional[str], use_same_token: bool = True) -> Dict[str, Any]:
    """
    Compare how highly error nodes rank in two graphs using a Mann–Whitney U test.
    Returns a dictionary with percentile ranks and statistical test results.
    """
    graph_metadata1 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                           prompt=prompt1)
    graph_metadata2 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                           prompt=prompt2)
    output_node1 = get_top_output_logit_node(graph_metadata1['nodes'])
    output_nodes2 = [node for node in get_output_logits(graph_metadata2['nodes']) if
                     node['node_id'].startswith(get_id_without_pos(output_node1['node_id']))]
    if use_same_token:
        assert len(output_nodes2) >= 1, (f"Can't find node corresponding with {output_node1['clerp']} in "
                                         f"second prompt.")
    else:
        output_nodes2 = [get_top_output_logit_node(graph_metadata2['nodes'])]
    linked_nodes1 = get_nodes_linked_to_target(graph_metadata1, output_node1)
    linked_nodes2 = get_nodes_linked_to_target(graph_metadata2, output_nodes2[0])
    results = compare_rankings(linked_nodes1, linked_nodes2)
    significant = []
    for metric, res in results.items():
        for key, val in res.items():
            if isinstance(val, dict):
                if val.get('p_value', 1) <= 0.05:
                    significant.append(key)
            elif key == 'p_value' and val <= 0.05:
                significant.append(metric)
    print(f"=========== {significant} ===========")
    return results


def compare_error_rankings(graph1_nodes: List[Dict[str, Any]],
                           graph2_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare how highly error nodes rank in two graphs using a Mann–Whitney U test.
    Each graph is a ranked list of node dictionaries.
    Nodes where node['feature_type'] == 'mlp reconstruction error' are treated as errors.
    """

    def get_error_percentile_ranks(graph_nodes: List[Dict[str, Any]]) -> List[float]:
        n = len(graph_nodes)
        error_percentiles = []

        for idx, node in enumerate(graph_nodes):
            if node.get("feature_type") == "mlp reconstruction error":
                # rank starts at 1 → idx starts at 0
                percentile = (idx + 1) / n
                error_percentiles.append(percentile)

        return error_percentiles

    # Extract percentile ranks for both graphs
    errors1 = get_error_percentile_ranks(graph1_nodes)
    errors2 = get_error_percentile_ranks(graph2_nodes)

    # Mann–Whitney requires at least 1 sample in each group
    if len(errors1) == 0 or len(errors2) == 0:
        raise ValueError("Both graphs must contain at least one error node.")

    # Run statistical test (alternative='two-sided' by default)
    u_stat, p_val = mannwhitneyu(errors1, errors2, alternative='two-sided')

    return {
        "percentile_ranks_1": errors1,
        "percentile_ranks_2": errors2,
        "u_statistic": u_stat,
        "p_value": p_val
    }


def get_relevance_list(graph_nodes: List[Dict[str, Any]]) -> List[int]:
    """
    Creates a binary relevance list where 1 indicates an error node and 0 indicates a non-error node.
    Used for computing ranking metrics like NDCG and average precision.
    """
    return [
        1 if node.get("feature_type") == "mlp reconstruction error" else 0
        for node in graph_nodes
    ]


def error_rank_mannwhitney(graph1_nodes: List[Dict[str, Any]],
                           graph2_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares the percentile ranks of error nodes between two graphs using Mann-Whitney U test.
    Returns percentile ranks for both graphs along with the U statistic and p-value.
    """

    def get_error_percentiles(nodes: List[Dict[str, Any]]) -> List[float]:
        n = len(nodes)
        return [
            (i + 1) / n
            for i, node in enumerate(nodes)
            if node.get("feature_type") == "mlp reconstruction error"
        ]

    p1 = get_error_percentiles(graph1_nodes)
    p2 = get_error_percentiles(graph2_nodes)

    if len(p1) == 0 or len(p2) == 0:
        raise ValueError("Both graphs must contain at least one error node.")

    u, p_val = mannwhitneyu(p1, p2, alternative="two-sided")

    return {
        "percentile_ranks_1": p1,
        "percentile_ranks_2": p2,
        "u_statistic": u,
        "p_value": p_val
    }


def top_k_error_proportion(graph_nodes: List[Dict[str, Any]], k: int) -> float:
    """
    Calculates the proportion of error nodes in the top k ranked nodes.
    Returns a value between 0 and 1 representing the error proportion.
    """
    rel = get_relevance_list(graph_nodes)
    k = min(k, len(rel))
    top_k = rel[:k]
    return sum(top_k) / k if k > 0 else 0.0


def ndcg_at_k(graph_nodes: List[Dict[str, Any]], k: int) -> float:
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) at rank k.
    NDCG measures the quality of ranking by comparing against an ideal ranking where all error nodes appear first.
    Returns a value between 0 and 1, where 1 indicates perfect ranking.
    """
    rel = get_relevance_list(graph_nodes)
    k = min(k, len(rel))
    rel_k = rel[:k]

    # DCG
    dcg = sum((rel_k[i] / np.log2(i + 2)) for i in range(k))

    # IDCG: ideal case (all error nodes at top)
    sorted_rel = sorted(rel, reverse=True)[:k]
    idcg = sum((sorted_rel[i] / np.log2(i + 2)) for i in range(k))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(graph_nodes: List[Dict[str, Any]]) -> float:
    """
    Calculates the average precision (AP) for error node ranking.
    AP summarizes the precision-recall curve by computing the average precision at each error node position.
    Returns a value between 0 and 1, where higher values indicate better ranking quality.
    """
    rel = get_relevance_list(graph_nodes)

    cum_precision = 0.0
    num_hits = 0

    for i, r in enumerate(rel):
        if r == 1:
            num_hits += 1
            cum_precision += num_hits / (i + 1)

    return cum_precision / num_hits if num_hits > 0 else 0.0


def extract_raw_values(pair_results, ks: List[int] = None):
    """
    Extracts the values for individual samples from all pair results.
    """
    if ks is None:
        ks = [5, 10, 20]

    g1 = {"top_k": {k: [] for k in ks}, "ndcg": {k: [] for k in ks}, "ap": []}
    g2 = {"top_k": {k: [] for k in ks}, "ndcg": {k: [] for k in ks}, "ap": []}

    for res in pair_results:

        # Top-K
        for k in ks:
            g1["top_k"][k].append(res["top_k_error_proportion"][f"top_{k}"]["metric_graph1"])
            g2["top_k"][k].append(res["top_k_error_proportion"][f"top_{k}"]["metric_graph2"])

        # NDCG
        for k in ks:
            g1["ndcg"][k].append(res["ndcg"][f"ndcg@{k}"]["metric_graph1"])
            g2["ndcg"][k].append(res["ndcg"][f"ndcg@{k}"]["metric_graph2"])

        # AP
        g1["ap"].append(res["average_precision"]["metric_graph1"])
        g2["ap"].append(res["average_precision"]["metric_graph2"])

    return g1, g2


def compare_rankings(graph1_nodes: List[Dict[str, Any]], graph2_nodes: List[Dict[str, Any]],
                     ks: Optional[List[int]] = None, n_permutations: int = 500) -> Dict[str, Any]:
    """
    Compares the ranking quality of error nodes between two graphs using multiple metrics.
    Computes Mann-Whitney U test, top-k error proportions, NDCG@k, and average precision.
    Returns a dictionary containing all computed metrics for both graphs.
    """
    if ks is None:
        ks = [5, 10, 20]

    results = {}

    # ---------------------------
    # Mann–Whitney U test
    # ---------------------------
    print(f"Running error test for mann-whitney.")
    results["mann_whitney"] = error_rank_mannwhitney(graph1_nodes, graph2_nodes)

    # ---------------------------
    # Metrics with permutation tests
    # ---------------------------
    results["top_k_error_proportion"] = {}
    results["ndcg"] = {}

    print(f"Running error analysis for topk and ndcg metrics")
    for k in ks:
        # top-k error proportion
        topk_fn = lambda g, k=k: top_k_error_proportion(g, k)

        results["top_k_error_proportion"][f"top_{k}"] = permutation_test_metric(
            graph1_nodes, graph2_nodes, topk_fn, n_permutations
        )

        # NDCG@k
        ndcg_fn = lambda g, k=k: ndcg_at_k(g, k)
        results["ndcg"][f"ndcg@{k}"] = permutation_test_metric(
            graph1_nodes, graph2_nodes, ndcg_fn, n_permutations
        )

    # ---------------------------
    # Average Precision
    # ---------------------------
    print(f"Running error test for average_precision.")
    results["average_precision"] = permutation_test_metric(
        graph1_nodes, graph2_nodes, average_precision, n_permutations
    )

    return results


def calculate_error_contributions(graph_metadata: dict, hops: int = 1) -> float:
    """
    Calculates the ratio of error contributions to overall.
    """
    node_dict = get_node_dict(graph_metadata)
    output_node = get_top_output_logit_node(graph_metadata['nodes'])
    links = get_links_from_node(graph_metadata, output_node, hops=hops)
    error_contribution = 0
    total_contribution = 0
    for link in links:
        source_id = link["source"]
        if node_dict[source_id].get("feature_type") == "mlp reconstruction error":
            error_contribution += 1
        total_contribution += 1
    return error_contribution / total_contribution


def run_error_analysis(prompts: List[str], model: str, submodel: str,
                       graph_dir: Optional[str]) -> Dict[str, float]:
    """
    Calculates the ratio of error contributions to total contributions for each prompt.
    Returns a dictionary mapping each prompt to its error contribution ratio.
    """
    metrics = {}
    for prompt in prompts:
        graph_metadata = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                              prompt=prompt)
        metrics[prompt] = calculate_error_contributions(graph_metadata)
    return metrics


def permutation_test_metric(graph1_nodes: List[Dict[str, Any]], graph2_nodes: List[Dict[str, Any]],
                            metric_fn: Any, n_permutations: int = 10000) -> Dict[str, float]:
    """
    Performs a permutation test for any metric that takes a ranked list of nodes and returns a scalar.
    The metric_fn must accept (graph_nodes) and return a float.
    Returns observed difference, p-value, and individual metric values for both graphs.
    """
    # observed difference
    m1 = metric_fn(graph1_nodes)
    m2 = metric_fn(graph2_nodes)
    observed_diff = m1 - m2

    # pool nodes
    pooled = graph1_nodes + graph2_nodes
    n1 = len(graph1_nodes)

    count = 0

    for _ in range(n_permutations):
        random.shuffle(pooled)

        # split into pseudo-graph1 and pseudo-graph2
        g1_perm = pooled[:n1]
        g2_perm = pooled[n1:]

        diff_perm = metric_fn(g1_perm) - metric_fn(g2_perm)

        if abs(diff_perm) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "metric_graph1": m1,
        "metric_graph2": m2
    }


def extract_metric_deltas(pair_results: List[Dict[str, Any]],
                          ks: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Extracts delta (difference) values from a list of compare_rankings() results.
    Computes Δ (graph1 - graph2) for top-k error proportion, NDCG@k, and average precision.
    Returns a dictionary of lists containing delta values for each metric.
    """
    if ks is None:
        ks = [5, 10, 20]

    deltas = {
        "top_k": {k: [] for k in ks},
        "ndcg": {k: [] for k in ks},
        "ap": []
    }

    for res in pair_results:

        # ---- Top-K
        for k in ks:
            d = res["top_k_error_proportion"][f"top_{k}"]["metric_graph1"] - \
                res["top_k_error_proportion"][f"top_{k}"]["metric_graph2"]
            deltas["top_k"][k].append(d)

        # ---- NDCG
        for k in ks:
            d = res["ndcg"][f"ndcg@{k}"]["metric_graph1"] - \
                res["ndcg"][f"ndcg@{k}"]["metric_graph2"]
            deltas["ndcg"][k].append(d)

        # ---- AP
        d = res["average_precision"]["metric_graph1"] - \
            res["average_precision"]["metric_graph2"]
        deltas["ap"].append(d)

    return deltas


def condition_level_stats(deltas: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test and paired t-test for each metric's delta values.
    Computes statistical tests, mean delta, and median delta for top-k, NDCG@k, and average precision.
    Returns a dictionary containing test results and summary statistics for each metric.
    """
    stats = {"top_k": {}, "ndcg": {}, "ap": {}}

    def compute_counts(vals):
        vals = np.array(vals)
        return (
            int(np.sum(vals > 0)),  # g1 > g2
            int(np.sum(vals < 0)),  # g1 < g2
            int(np.sum(vals == 0)),  # tie
        )

    # ---- Top-K
    for k, vals in deltas["top_k"].items():
        vals = np.array(vals)
        w = wilcoxon(vals)
        t = ttest_rel(vals, np.zeros_like(vals))
        c1_gt, c1_lt, c_eq = compute_counts(vals)

        stats["top_k"][k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "ttest_stat": float(t.statistic),
            "ttest_p": float(t.pvalue),
            "count_g1_gt_g2": c1_gt,
            "count_g1_lt_g2": c1_lt,
            "count_equal": c_eq,
        }

    # ---- NDCG
    for k, vals in deltas["ndcg"].items():
        vals = np.array(vals)
        w = wilcoxon(vals)
        t = ttest_rel(vals, np.zeros_like(vals))
        c1_gt, c1_lt, c_eq = compute_counts(vals)

        stats["ndcg"][k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "ttest_stat": float(t.statistic),
            "ttest_p": float(t.pvalue),
            "count_g1_gt_g2": c1_gt,
            "count_g1_lt_g2": c1_lt,
            "count_equal": c_eq,
        }

    # ---- AP
    vals = np.array(deltas["ap"])
    w = wilcoxon(vals)
    t = ttest_rel(vals, np.zeros_like(vals))
    c1_gt, c1_lt, c_eq = compute_counts(vals)

    stats["ap"] = {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "ttest_stat": float(t.statistic),
        "ttest_p": float(t.pvalue),
        "count_g1_gt_g2": c1_gt,
        "count_g1_lt_g2": c1_lt,
        "count_equal": c_eq,
    }

    return stats


def pooled_condition_stats(all_deltas: Dict[str, Dict[str, Any]],
                            ks: Optional[List[int]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Pools delta values across all conditions and computes summary statistics.
    Uses one-sided Wilcoxon signed-rank test (alternative="greater") to determine
    whether the base condition consistently outperforms others.

    Args:
        all_deltas: Dictionary mapping condition_name -> deltas dict
        ks: List of k values for top-k and NDCG metrics

    Returns:
        Tuple of (pooled_stats, pooled_deltas)
    """
    if ks is None:
        ks = [5, 10, 20]

    # Initialize pooled deltas
    pooled_deltas = {
        "top_k": {k: [] for k in ks},
        "ndcg": {k: [] for k in ks},
        "ap": []
    }

    # Pool deltas from all conditions
    for condition_name, deltas in all_deltas.items():
        for k in ks:
            pooled_deltas["top_k"][k].extend(deltas["top_k"][k])
            pooled_deltas["ndcg"][k].extend(deltas["ndcg"][k])
        pooled_deltas["ap"].extend(deltas["ap"])

    # Compute stats with one-sided Wilcoxon test
    stats = {"top_k": {}, "ndcg": {}, "ap": {}}

    def compute_counts(vals):
        vals = np.array(vals)
        return (
            int(np.sum(vals > 0)),  # base > other
            int(np.sum(vals < 0)),  # base < other
            int(np.sum(vals == 0)),  # tie
        )

    # ---- Top-K
    for k, vals in pooled_deltas["top_k"].items():
        vals = np.array(vals)
        # One-sided test: is base condition greater?
        try:
            w = wilcoxon(vals, alternative="greater")
            wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            # All values might be zero
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        c1_gt, c1_lt, c_eq = compute_counts(vals)

        stats["top_k"][k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "wilcoxon_stat_onesided": wilcoxon_stat,
            "wilcoxon_p_onesided": wilcoxon_p,
            "count_base_gt_other": c1_gt,
            "count_base_lt_other": c1_lt,
            "count_equal": c_eq,
            "n_samples": len(vals),
        }

    # ---- NDCG
    for k, vals in pooled_deltas["ndcg"].items():
        vals = np.array(vals)
        try:
            w = wilcoxon(vals, alternative="greater")
            wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        c1_gt, c1_lt, c_eq = compute_counts(vals)

        stats["ndcg"][k] = {
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "wilcoxon_stat_onesided": wilcoxon_stat,
            "wilcoxon_p_onesided": wilcoxon_p,
            "count_base_gt_other": c1_gt,
            "count_base_lt_other": c1_lt,
            "count_equal": c_eq,
            "n_samples": len(vals),
        }

    # ---- AP
    vals = np.array(pooled_deltas["ap"])
    try:
        w = wilcoxon(vals, alternative="greater")
        wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
    except ValueError:
        wilcoxon_stat, wilcoxon_p = np.nan, np.nan

    c1_gt, c1_lt, c_eq = compute_counts(vals)

    stats["ap"] = {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "wilcoxon_stat_onesided": wilcoxon_stat,
        "wilcoxon_p_onesided": wilcoxon_p,
        "count_base_gt_other": c1_gt,
        "count_base_lt_other": c1_lt,
        "count_equal": c_eq,
        "n_samples": len(vals),
    }

    return stats, pooled_deltas


def plot_delta_distribution(delta_values: List[float], title: str,
                            save_path: Optional[Path] = None) -> None:
    """
    Creates a histogram plot showing the distribution of delta values.
    Visualizes the density distribution of metric differences between two conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create histogram with KDE overlay
    sns.histplot(delta_values, bins=15, kde=True, color=COLORS['turquoise'], alpha=0.7,
                 edgecolor='white', linewidth=0.8, ax=ax)

    # Add vertical line at zero for reference
    ax.axvline(x=0, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')

    # Add mean line
    mean_val = np.mean(delta_values)
    ax.axvline(x=mean_val, color=COLORS['pastel_orange'], linestyle='-',
               linewidth=2, alpha=0.9, label=f'Mean: {mean_val:.3f}')

    ax.set_title(title, pad=15)
    ax.set_xlabel(f"Δ ({CONDITION1} - {CONDITION2})", labelpad=10)
    ax.set_ylabel("Density", labelpad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def boxplot_metric_family(metric_dict: Dict[int, List[float]], title_prefix: str,
                          save_path: Optional[Path] = None) -> None:
    """
    Creates boxplot visualizations for a group of related metrics across different K values.
    The metric_dict should map K values to lists of delta values.
    Useful for visualizing NDCG@k or top-k error proportion across different k values.
    """
    ks = sorted(metric_dict.keys())

    # Prepare data in long format for seaborn
    plot_data = []
    for k in ks:
        for val in metric_dict[k]:
            plot_data.append({'K': f'K={k}', 'Delta': val})
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create boxplot with strip plot overlay
    sns.boxplot(x='K', y='Delta', data=df, palette=CUSTOM_PALETTE, width=0.5,
                linewidth=1.5, fliersize=0, ax=ax)
    sns.stripplot(x='K', y='Delta', data=df, color='#333333', alpha=0.5,
                  size=4, jitter=0.15, ax=ax)

    # Add horizontal line at zero
    ax.axhline(y=0, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title(f"{title_prefix}: Δ Distributions", pad=15)
    ax.set_xlabel("Top-K Value", labelpad=10)
    ax.set_ylabel(f"Δ ({CONDITION1} - {CONDITION2})", labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def stats_to_csv(stats: dict[str, Any], filepath: Path) -> pd.DataFrame:
    """
    Saves stats from error analysis to csv.
    """
    rows = []

    # Top-K
    for k, s in stats["top_k"].items():
        rows.append({
            "metric": f"top_k_{k}",
            **s
        })

    # NDCG
    for k, s in stats["ndcg"].items():
        rows.append({
            "metric": f"ndcg_{k}",
            **s
        })

    # AP
    rows.append({
        "metric": "average_precision",
        **stats["ap"]
    })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df


def raw_scores_to_csv(g1_values: Dict[str, Any], g2_values: Dict[str, Any],
                      deltas: Dict[str, Any], filepath: Path,
                      sample_ids: List[str] = None):
    """
    Saves the sample scores to a csv.
    """
    rows = []

    # -------- Top-K --------
    for k in g1_values["top_k"]:
        g1_list = g1_values["top_k"][k]
        g2_list = g2_values["top_k"][k]
        d_list = deltas["top_k"][k]

        for i, (g1, g2, d) in enumerate(zip(g1_list, g2_list, d_list)):
            sample_id = i if not sample_ids else sample_ids[i]
            rows.append({
                "metric": f"top_k_{k}",
                "sample": sample_id,
                "g1_score": g1,
                "g2_score": g2,
                "delta": d
            })

    # -------- NDCG --------
    for k in g1_values["ndcg"]:
        g1_list = g1_values["ndcg"][k]
        g2_list = g2_values["ndcg"][k]
        d_list = deltas["ndcg"][k]

        for i, (g1, g2, d) in enumerate(zip(g1_list, g2_list, d_list)):
            sample_id = i if not sample_ids else sample_ids[i]
            rows.append({
                "metric": f"ndcg_{k}",
                "sample": sample_id,
                "g1_score": g1,
                "g2_score": g2,
                "delta": d
            })

    # -------- AP --------
    for i, (g1, g2, d) in enumerate(zip(g1_values["ap"], g2_values["ap"], deltas["ap"])):
        sample_id = i if not sample_ids else sample_ids[i]
        rows.append({
            "metric": "average_precision",
            "sample": sample_id,
            "g1_score": g1,
            "g2_score": g2,
            "delta": d
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return df


def analyze_conditions(pair_results: dict[str, Dict[str, Dict[str, Any]]],
                       ks: Optional[List[int]] = None,
                       save_path: Path = None) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Performs comprehensive statistical analysis and visualization of error ranking metrics.
    Extracts delta values, computes statistical tests, and generates visualizations.
    Also pools results across all conditions to evaluate whether the base condition
    consistently outperforms all others.

    Args:
        pair_results: Dictionary mapping sample_id -> condition_name -> results_dict
        ks: List of k values for top-k and NDCG metrics
        save_path: Base path for saving results (subdirs created per condition)

    Returns:
        Dictionary mapping condition_name -> (stats, deltas) tuple
        Includes special key "pooled" for pooled analysis across all conditions
    """
    if ks is None:
        ks = [5, 10, 20]

    sample_ids = list(pair_results.keys())

    # Get all condition names from the first sample
    first_sample = next(iter(pair_results.values()))
    condition_names = list(first_sample.keys())

    all_results = {}
    all_deltas = {}

    for condition_name in condition_names:
        # Extract results for this condition across all samples
        condition_results = [pair_results[sample_id][condition_name] for sample_id in sample_ids]

        deltas = extract_metric_deltas(condition_results, ks)
        stats = condition_level_stats(deltas)
        g1_values, g2_values = extract_raw_values(condition_results)

        if save_path:
            # Create condition subdirectory
            condition_save_path = save_path / condition_name
            condition_save_path.mkdir(parents=True, exist_ok=True)

            raw_scores_to_csv(g1_values, g2_values, deltas, sample_ids=sample_ids,
                              filepath=condition_save_path / "error-results-individual.csv")

            # Save plots to files
            boxplot_metric_family(deltas["top_k"], "Top-K Error Proportion",
                                  save_path=condition_save_path / "topk_boxplot.png")
            boxplot_metric_family(deltas["ndcg"], "NDCG@K",
                                  save_path=condition_save_path / "ndcg_boxplot.png")
            plot_delta_distribution(deltas["ap"], "Average Precision Δ Distribution",
                                    save_path=condition_save_path / "ap_distribution.png")

            # Save stats to CSV
            stats_to_csv(stats, filepath=condition_save_path / "error-results.csv")
            print(f"Saved error results for condition '{condition_name}' to: {condition_save_path}")

        all_results[condition_name] = (stats, deltas)
        all_deltas[condition_name] = deltas

    # Pooled analysis across all conditions
    pooled_stats, pooled_deltas = pooled_condition_stats(all_deltas, ks)

    if save_path:
        # Create pooled subdirectory
        pooled_save_path = save_path / "pooled"
        pooled_save_path.mkdir(parents=True, exist_ok=True)

        # Save plots for pooled results
        boxplot_metric_family(pooled_deltas["top_k"], "Top-K Error Proportion (Pooled)",
                              save_path=pooled_save_path / "topk_boxplot.png")
        boxplot_metric_family(pooled_deltas["ndcg"], "NDCG@K (Pooled)",
                              save_path=pooled_save_path / "ndcg_boxplot.png")
        plot_delta_distribution(pooled_deltas["ap"], "Average Precision Δ Distribution (Pooled)",
                                save_path=pooled_save_path / "ap_distribution.png")

        # Save pooled stats to CSV
        stats_to_csv(pooled_stats, filepath=pooled_save_path / "error-results.csv")
        print(f"Saved pooled error results to: {pooled_save_path}")

    all_results["pooled"] = (pooled_stats, pooled_deltas)

    return all_results
