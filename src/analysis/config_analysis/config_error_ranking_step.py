import random
from collections import namedtuple
from enum import Enum
from typing import Dict, List, Any, Optional, Set

import numpy as np
from scipy.stats import mannwhitneyu

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.graph_manager import GraphManager
from src.utils import Metrics


class ErrorRankingMetrics(Metrics):
    MANN_WHITNEY = "mann_whitney"
    TOP_K = "top_k_error_proportion"
    NDCG = "ndcg"
    AP = "average_precision"


MannWhitneyResult = namedtuple("MannWhitneyResult",
                               ["percentile_ranks_1", "percentile_ranks_2", "u_statistic", "p_value"])

PermutationTestResult = namedtuple("PermutationTestResult",
                                   ["observed_diff", "p_value", "metric_graph1", "metric_graph2"])


class ConfigErrorRankingStep(ConfigAnalyzeStep):
    DEFAULT_Ks = [5, 10, 20]
    ALL_METRICS = [ErrorRankingMetrics.TOP_K, ErrorRankingMetrics.NDCG, ErrorRankingMetrics.AP]
    K_METRICS = [ErrorRankingMetrics.TOP_K, ErrorRankingMetrics.NDCG]

    def __init__(self, metrics2run: Set[ErrorRankingMetrics] = None, use_same_token: bool = True):
        """
        Initializes the config error ranking step.

        Args:
            metrics2run: Set of metrics to compute.
            use_same_token: Whether to use the same output token for both graphs.
        """
        self.metrics2run = {e for e in ErrorRankingMetrics} if not metrics2run else metrics2run
        self.use_same_token = use_same_token
        self.metric_fns = {
            ErrorRankingMetrics.TOP_K: self.top_k_error_proportion,
            ErrorRankingMetrics.NDCG: self.ndcg_at_k,
            ErrorRankingMetrics.AP: self.average_precision,
        }
        super().__init__()

    def run(self, graphs: List[GraphManager], conditions: List[str] = None) -> Dict:
        """
        Runs the error ranking analysis on the provided graphs.

        Args:
            graphs: List of graphs to run analysis on.
            conditions: Conditions corresponding with each graph.


        Returns:
            Dictionary of results.
        """
        return self._run_error_ranking(graphs, self.use_same_token)

    def _run_error_ranking(self, graphs: List[GraphManager], use_same_token: bool = True) -> Dict:
        """
        Compares error node rankings between two graphs.

        Args:
            graphs: List of graphs to run analysis on.
            use_same_token: Whether to use the same output token for both graphs.

        Returns:
            Dictionary of results.
        """
        assert len(graphs) == 2, f"Expected two graphs for error ranking analysis. Received {len(graphs)}"
        graph1, graph2 = graphs
        output_node1 = graph1.get_top_output_logit_node()
        if use_same_token:
            output_node2 = graph2.find_output_node(output_node1)
        else:
            output_node2 = graph2.get_top_output_logit_node()
        linked_nodes1 = graph1.get_nodes_linked_to_target(output_node1)
        linked_nodes2 = graph2.get_nodes_linked_to_target(output_node2)
        results = self.compare_rankings(linked_nodes1, linked_nodes2)
        return results

    def compare_rankings(self, nodes1: List[Dict], nodes2: List[Dict],
                         ks: Optional[List[int]] = None, n_permutations: int = 500) -> Dict[str, Any]:
        """
        Compares error node ranking quality between two graphs using multiple metrics.

        Args:
            nodes1: Ranked list of nodes from graph 1.
            nodes2: Ranked list of nodes from graph 2.
            ks: List of k values for top-k and NDCG metrics.
            n_permutations: Number of permutations for significance testing.

        Returns:
            Dictionary of computed metrics for both graphs.
        """
        ks = self.DEFAULT_Ks if ks is None else ks
        results = {}

        # Handle Mann-Whitney separately (doesn't use permutation test)
        if ErrorRankingMetrics.MANN_WHITNEY in self.metrics2run:
            print(f"Running error test for {ErrorRankingMetrics.MANN_WHITNEY.value}.")
            results[ErrorRankingMetrics.MANN_WHITNEY.value] = self.error_rank_mannwhitney(nodes1, nodes2)

        # Process permutation test metrics
        for metric in self.ALL_METRICS:
            if metric not in self.metrics2run:
                continue

            base_fn = self.metric_fns[metric]

            if metric in self.K_METRICS:
                results[metric.value] = {}
                for k in ks:
                    print(f"Running error analysis for {metric.value} @ {k}")
                    metric_fn = lambda g, k=k: base_fn(g, k)
                    results[metric.value][k] = self.permutation_test_metric(
                        nodes1, nodes2, metric_fn, n_permutations
                    )
            else:
                print(f"Running error test for {metric.value}.")
                results[metric.value] = self.permutation_test_metric(
                    nodes1, nodes2, base_fn, n_permutations
                )

        return results

    @staticmethod
    def error_rank_mannwhitney(nodes1: List[Dict], nodes2: List[Dict]) -> MannWhitneyResult:
        """
        Compares error node percentile ranks between two graphs using Mann-Whitney U test.

        Args:
            nodes1: Ranked list of nodes from graph 1.
            nodes2: Ranked list of nodes from graph 2.

        Returns:
            MannWhitneyResult with percentile ranks and test statistics.
        """
        errors1 = ConfigErrorRankingStep.get_error_percentile_ranks(nodes1)
        errors2 = ConfigErrorRankingStep.get_error_percentile_ranks(nodes2)

        # Mann–Whitney requires at least 1 sample in each group
        if len(errors1) == 0 or len(errors2) == 0:
            raise ValueError("Both graphs must contain at least one error node.")

        u_stat, p_val = mannwhitneyu(errors1, errors2, alternative='two-sided')

        return MannWhitneyResult(percentile_ranks_1=errors1,
                                 percentile_ranks_2=errors2,
                                 u_statistic=u_stat,
                                 p_value=p_val)

    @staticmethod
    def ndcg_at_k(nodes: List[Dict[str, Any]], k: int) -> float:
        """
        Calculates Normalized Discounted Cumulative Gain (NDCG) at rank k.

        Args:
            nodes: Ranked list of node dictionaries.
            k: Number of top positions to consider.

        Returns:
            NDCG score between 0 and 1.
        """
        rel = ConfigErrorRankingStep.get_relevance_list(nodes)
        k = min(k, len(rel))
        rel_k = rel[:k]

        # DCG
        dcg = sum((rel_k[i] / np.log2(i + 2)) for i in range(k))

        # IDCG: ideal case (all error nodes at top)
        sorted_rel = sorted(rel, reverse=True)[:k]
        idcg = sum((sorted_rel[i] / np.log2(i + 2)) for i in range(k))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def average_precision(nodes: List[Dict[str, Any]]) -> float:
        """
        Calculates average precision (AP) for error node ranking.

        Args:
            nodes: Ranked list of node dictionaries.

        Returns:
            AP score between 0 and 1.
        """
        rel = ConfigErrorRankingStep.get_relevance_list(nodes)

        cum_precision = 0.0
        num_hits = 0

        for i, r in enumerate(rel):
            if r == 1:
                num_hits += 1
                cum_precision += num_hits / (i + 1)

        return cum_precision / num_hits if num_hits > 0 else 0.0

    @staticmethod
    def permutation_test_metric(nodes1: List[Dict], nodes2: List[Dict],
                                metric_fn: Any, n_permutations: int = 10000) -> PermutationTestResult:
        """
        Performs a permutation test for a given metric function.

        Args:
            nodes1: Ranked list of nodes from graph 1.
            nodes2: Ranked list of nodes from graph 2.
            metric_fn: Function that takes nodes and returns a scalar.
            n_permutations: Number of permutations to run.

        Returns:
            PermutationTestResult with observed difference, p-value, and metric values.
        """
        # observed difference
        m1 = metric_fn(nodes1)
        m2 = metric_fn(nodes2)
        observed_diff = m1 - m2

        # pool nodes
        pooled = nodes1 + nodes2
        n1 = len(nodes1)

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
        return PermutationTestResult(observed_diff=observed_diff, p_value=p_value,
                                     metric_graph1=m1, metric_graph2=m2)

    @staticmethod
    def get_relevance_list(nodes: List[Dict[str, Any]]) -> List[int]:
        """
        Creates a binary relevance list (1 for error nodes, 0 otherwise).

        Args:
            nodes: List of node dictionaries.

        Returns:
            List of binary relevance values.
        """
        return [
            1 if ConfigErrorRankingStep.is_error_node(node) else 0
            for node in nodes
        ]

    @staticmethod
    def is_error_node(node: dict[str, Any]) -> bool | Any:
        """
        Checks if a node is an MLP reconstruction error node.

        Args:
            node: Node dictionary.

        Returns:
            True if node is an error node.
        """
        return node.get("feature_type") == "mlp reconstruction error"

    @staticmethod
    def get_error_percentile_ranks(nodes: List[Dict[str, Any]]) -> List[float]:
        """
        Computes percentile ranks of error nodes in the ranking.

        Args:
            nodes: Ranked list of node dictionaries.

        Returns:
            List of percentile ranks for error nodes.
        """
        n = len(nodes)
        error_percentiles = []

        for idx, node in enumerate(nodes):
            if ConfigErrorRankingStep.is_error_node(node):
                percentile = (idx + 1) / n
                error_percentiles.append(percentile)

        return error_percentiles

    @staticmethod
    def top_k_error_proportion(nodes: List[Dict[str, Any]], k: int) -> float:
        """
        Calculates the proportion of error nodes in the top k positions.

        Args:
            nodes: Ranked list of node dictionaries.
            k: Number of top positions to consider.

        Returns:
            Error proportion between 0 and 1.
        """
        rel = ConfigErrorRankingStep.get_relevance_list(nodes)
        k = min(k, len(rel))
        top_k = rel[:k]
        return sum(top_k) / k if k > 0 else 0.0
