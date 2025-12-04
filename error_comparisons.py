import random
from typing import Optional, List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, ttest_rel

from attribution_graph_utils import create_or_load_graph, get_top_output_logit_node, get_output_logits, \
    get_nodes_linked_to_target, get_node_dict, get_links_from_node
from common_utils import get_id_without_pos


def run_error_ranking(prompt1: str, prompt2: str, model: str, submodel: str,
                      graph_dir: Optional[str]) -> Dict[str, Any]:
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
    assert len(output_nodes2) >= 1, (f"Can't find node corresponding with {output_node1['clerp']} in "
                                     f"second prompt.")
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

    # ---- Top-K
    for k, vals in deltas["top_k"].items():
        stats["top_k"][k] = {
            "wilcoxon": wilcoxon(vals, alternative='two-sided'),
            "ttest": ttest_rel(vals, np.zeros_like(vals)),
            "mean_delta": np.mean(vals),
            "median_delta": np.median(vals),
        }

    # ---- NDCG
    for k, vals in deltas["ndcg"].items():
        stats["ndcg"][k] = {
            "wilcoxon": wilcoxon(vals, alternative='two-sided'),
            "ttest": ttest_rel(vals, np.zeros_like(vals)),
            "mean_delta": np.mean(vals),
            "median_delta": np.median(vals),
        }

    # ---- AP
    vals = deltas["ap"]
    stats["ap"] = {
        "wilcoxon": wilcoxon(vals, alternative='two-sided'),
        "ttest": ttest_rel(vals, np.zeros_like(vals)),
        "mean_delta": np.mean(vals),
        "median_delta": np.median(vals),
    }

    return stats


def plot_delta_distribution(delta_values: List[float], title: str) -> None:
    """
    Creates a histogram plot showing the distribution of delta values.
    Visualizes the density distribution of metric differences between two conditions.
    """
    plt.figure(figsize=(6,4))
    plt.hist(delta_values, bins=20, alpha=0.6, density=True)
    plt.title(title)
    plt.xlabel("Δ (metric1 - metric2)")
    plt.ylabel("Density")
    plt.show()


def boxplot_metric_family(metric_dict: Dict[int, List[float]], title_prefix: str) -> None:
    """
    Creates boxplot visualizations for a group of related metrics across different K values.
    The metric_dict should map K values to lists of delta values.
    Useful for visualizing NDCG@k or top-k error proportion across different k values.
    """
    ks = sorted(metric_dict.keys())
    data = [metric_dict[k] for k in ks]

    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=[str(k) for k in ks])
    plt.title(f"{title_prefix}: Δ distributions")
    plt.xlabel("K")
    plt.ylabel("Δ (metric1 - metric2)")
    plt.show()


def analyze_condition(pair_results: List[Dict[str, Any]],
                     ks: Optional[List[int]] = None, show_plots: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Performs comprehensive statistical analysis and visualization of error ranking metrics.
    Extracts delta values, computes statistical tests, and generates visualizations.
    Returns statistics dictionary and deltas dictionary.
    """
    if ks is None:
        ks = [5, 10, 20]

    deltas = extract_metric_deltas(pair_results, ks)
    stats = condition_level_stats(deltas)

    # ----- Visualizations -----

    if show_plots:
        # Top-K boxplot
        boxplot_metric_family(deltas["top_k"], "Top-K Error Proportion")

        # NDCG boxplot
        boxplot_metric_family(deltas["ndcg"], "NDCG@K")

        # AP distribution
        plot_delta_distribution(deltas["ap"], "Average Precision Δ Distribution")

    return stats, deltas