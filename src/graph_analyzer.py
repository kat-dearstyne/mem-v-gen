import enum
from collections import namedtuple
from typing import Dict, Tuple, Any, List, Set, Type

import numpy as np
import pandas as pd
from numpy import ndarray, dtype

from src.constants import MIN_ACTIVATION_DENSITY
from src.graph_manager import GraphManager
from src.neuronpedia_manager import NeuronpediaManager
from src.utils import Metrics


class ComparisonMetrics(Metrics):
    JACCARD_INDEX = 'jaccard_index'
    WEIGHTED_JACCARD = 'weighted_jaccard'
    FRAC_FROM_INTERSECTION = 'frac_from_intersection'
    SHARED_TOKEN = 'shared_token'
    OUTPUT_PROBABILITY = 'output_probability'


class SharedFeatureMetrics(Metrics):
    NUM_SHARED = 'num_shared'
    NUM_PROMPTS = 'num_prompts'
    AVG_FEATURES_PER_PROMPT = 'avg_features_per_prompt'
    COUNT_AT_THRESHOLD = 'count_at_{}pct'
    SHARED_PRESENT_PER_PROMPT = 'shared_present_per_prompt'


class SubgraphComparisonMetrics(Metrics):
    """Top-level categories for subgraph comparison results."""
    INTERSECTION_METRICS = 'intersection_metrics'
    FEATURE_PRESENCE = 'feature_presence'
    SHARED_FEATURES = 'shared_features'


class GraphAnalyzer:

    DEFAULT_THRESHOLDS = [50, 75, 100]
    PRIMARY_THRESHOLD = 50

    def __init__(self, prompts: Dict[str, str], neuronpedia_manager: NeuronpediaManager):
        """
        Initialize a GraphAnalyzer for comparing attribution graphs across prompts.

        Args:
            prompts: Dictionary mapping prompt IDs to prompt strings.
            neuronpedia_manager: NeuronpediaManager instance for loading graphs.
        """
        self.prompts = prompts
        self.neuronpedia_manager = neuronpedia_manager
        self.graphs, self.dfs = self.load_graphs_and_dfs()

    def load_graphs_and_dfs(self) -> Tuple[Dict[str, GraphManager], Dict[str, pd.DataFrame]]:
        """
        Loads all graphs and creates node dataframes once for reuse across methods.

        Returns:
            Tuple of (graphs dict, dataframes dict) keyed by prompt ID.
        """
        graphs = {}
        dfs = {}
        for p_id, prompt in self.prompts.items():
            graph = self.neuronpedia_manager.create_or_load_graph(prompt=prompt)
            df = graph.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                      drop_duplicates=True)
            graphs[p_id] = graph
            dfs[p_id] = df
        return graphs, dfs

    def get_graph_and_df(self, prompt_id: str):
        """
        Retrieve the graph and node dataframe for a given prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            Tuple of (GraphManager, DataFrame) for the prompt.
        """
        assert prompt_id in self.prompts, f"Unknown prompt: {prompt_id}"
        return self.graphs[prompt_id], self.dfs[prompt_id]

    def _compare_nodes(
            self,
            main_prompt_id: str,
            comparison_prompts: List[str] | None,
            combine_fn,
            verbose: bool = False,
            raise_if_no_matching_tokens: bool = False,
            metrics2run: Set[Metrics] | str = None
    ) -> Tuple[pd.DataFrame, dict[str, Any]] | pd.DataFrame:
        """
        Generic method for comparing nodes across prompts.

        Args:
            main_prompt_id: The main prompt to compare from.
            comparison_prompts: List of prompt IDs to compare against.
            combine_fn: Function(current_df, comparison_df) -> new_df.
            verbose: Whether to print verbose output.
            raise_if_no_matching_tokens: Whether to raise if tokens don't match.
            metrics2run: Metrics to calculate.

        Returns:
            DataFrame of resulting features, or tuple of (DataFrame, metrics dict) if metrics2run is set.
        """
        assert main_prompt_id in self.prompts, f"Unknown prompt: {main_prompt_id}"
        comparison_prompts = list(self.prompts.keys()) if not comparison_prompts else comparison_prompts
        graph1, node_df1 = self.get_graph_and_df(main_prompt_id)
        result_df = node_df1.copy()
        metric_results = {}

        if verbose:
            output_node = graph1.get_top_output_logit_node()
            print(f"Main prompt output: ", output_node['clerp'])

        for prompt_id in self.prompts.keys():
            if prompt_id == main_prompt_id or prompt_id not in comparison_prompts:
                continue
            _, node_df2 = self.get_graph_and_df(prompt_id)
            result_df = combine_fn(result_df, node_df2)

            if metrics2run:
                intersection_metrics = self.calculate_intersection_metrics(
                        prompt1_id=main_prompt_id, prompt2_id=prompt_id, metrics=metrics2run,
                        verbose=verbose, raise_if_no_matching_tokens=raise_if_no_matching_tokens
                    )
                if intersection_metrics:
                    metric_results[prompt_id] = intersection_metrics

        if metrics2run:
            shared_features = self.calculate_shared_feature_metrics(
                comparison_prompts=comparison_prompts
            )
            if shared_features:
                metric_results["shared_features"] = shared_features
            return result_df, metric_results
        return result_df

    def nodes_not_in(self, main_prompt_id: str,
                     comparison_prompts: List[str] | None = None,
                     verbose: bool = False,
                     raise_if_no_matching_tokens: bool = False,
                     metrics2run: Set[Metrics] | str = None):
        """
        Returns nodes from main_prompt that are NOT in any of the comparison prompts.

        Args:
            main_prompt_id: The main prompt to compare from.
            comparison_prompts: List of prompt IDs to compare against. Defaults to all prompts.
            verbose: Whether to print verbose output.
            raise_if_no_matching_tokens: Whether to raise if tokens don't match.
            metrics2run: Set of metrics to calculate, or 'all'.

        Returns:
            DataFrame of unique features, or tuple of (DataFrame, metrics dict) if metrics2run is set.
        """
        return self._compare_nodes(
            main_prompt_id=main_prompt_id,
            comparison_prompts=comparison_prompts,
            combine_fn=self.filter_for_unique_features,
            verbose=verbose,
            raise_if_no_matching_tokens=raise_if_no_matching_tokens,
            metrics2run=metrics2run
        )

    def nodes_in(self, main_prompt_id: str,
                 comparison_prompts: List[str] | None = None,
                 verbose: bool = False,
                 raise_if_no_matching_tokens: bool = False,
                 metrics2run: Set[Metrics] | str = None) -> Tuple[
                                                                          pd.DataFrame, dict[str, Any]] | pd.DataFrame:
        """
        Compares the nodes across prompts and returns a dataframe of only the nodes which are shared across all prompts.

        Args:
            main_prompt_id: The main prompt to compare from.
            comparison_prompts: List of prompt IDs to compare against. Defaults to all prompts.
            verbose: Whether to print verbose output.
            raise_if_no_matching_tokens: Whether to raise if tokens don't match.
            metrics2run: Set of metrics to calculate, or 'all'.

        Returns:
            DataFrame of shared features, or tuple of (DataFrame, metrics dict) if metrics2run is set.
        """

        return self._compare_nodes(
            main_prompt_id=main_prompt_id,
            comparison_prompts=comparison_prompts,
            combine_fn=self.intersect_features,
            verbose=verbose,
            raise_if_no_matching_tokens=raise_if_no_matching_tokens,
            metrics2run=metrics2run
        )

    def filter_for_unique_features(self, node_df1: pd.DataFrame, node_df2: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out features that are in node_df2 and keeps only features unique to node_df1.

        Args:
            node_df1: DataFrame of features to filter.
            node_df2: DataFrame of features to exclude.

        Returns:
            DataFrame containing only features from node_df1 that are not in node_df2.
        """
        diff = (
            node_df1
            .merge(node_df2[GraphManager.NODE_COLUMNS].drop_duplicates(), on=self.COLUMNS, how="left",
                   indicator=True)
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )
        return diff

    def intersect_features(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Returns features that exist in both dataframes.

        Args:
            df1: First DataFrame of features.
            df2: Second DataFrame of features.

        Returns:
            DataFrame containing only features present in both inputs.
        """
        merged = pd.merge(df1, df2[GraphManager.NODE_COLUMNS].drop_duplicates(), on=self.COLUMNS, how='inner')
        return merged[GraphManager.NODE_COLUMNS].drop_duplicates()

    def calculate_intersection_metrics(self,
                                       prompt1_id: str,
                                       prompt2_id: str,
                                       metrics: Set[ComparisonMetrics] | str = 'all',
                                       verbose: bool = False,
                                       raise_if_no_matching_tokens: bool = True) -> dict[str, float] | None:
        """
        Calculates intersection metrics between two prompts' graphs.

        Args:
            prompt1_id: The first prompt identifier.
            prompt2_id: The second prompt identifier.
            metrics: Set of ComparisonMetrics to calculate, or 'all'.
            verbose: Whether to print verbose output.
            raise_if_no_matching_tokens: Whether to raise if output tokens don't match.

        Returns:
            Dictionary mapping metric names to values, or None if no valid metrics requested.
        """
        metrics = {e for e in ComparisonMetrics} if metrics == "all" else metrics
        metrics = {metrics for metric in metrics if isinstance(metric, ComparisonMetrics)}
        if not metrics:
            return None
        metric_results = {}
        graph1, node_df1 = self.get_graph_and_df(prompt1_id)
        graph2, node_df2 = self.get_graph_and_df(prompt2_id)
        intersection = self.get_intersection(node_df1, node_df2)
        links_lookup, _, output_node2 = self.get_links_overlap(graph1, graph2,
                                                               raise_if_no_matching_tokens=raise_if_no_matching_tokens)
        if verbose:
            print(f"Prompt {prompt2_id} output: ", graph2['clerp'])
            print(f"Graph URL: {graph2.url}")

        if ComparisonMetrics.WEIGHTED_JACCARD in metrics:
            metric_results[ComparisonMetrics.WEIGHTED_JACCARD] = self.calculated_weighted_jaccard(links_lookup)
        if ComparisonMetrics.JACCARD_INDEX in metrics:
            metric_results[ComparisonMetrics.JACCARD_INDEX] = self.calculated_jaccard_index(node_df1, node_df2,
                                                                                            intersection)
        if ComparisonMetrics.FRAC_FROM_INTERSECTION:
            metric_results[ComparisonMetrics.FRAC_FROM_INTERSECTION] = len(intersection) / len(node_df1)
        if ComparisonMetrics.OUTPUT_PROBABILITY:
            metric_results[ComparisonMetrics.OUTPUT_PROBABILITY] = output_node2['token_prob']
        if ComparisonMetrics.SHARED_TOKEN:
            metric_results[ComparisonMetrics.SHARED_TOKEN] = output_node2['clerp']
        return {metric.value: res for metric, res in metric_results.items()}

    def calculated_jaccard_index(self, node_df1: pd.DataFrame, node_df2: pd.DataFrame,
                                 intersection: pd.Series) -> float:
        """
        Calculate the Jaccard index between two node dataframes.

        Args:
            node_df1: First DataFrame of nodes.
            node_df2: Second DataFrame of nodes.
            intersection: Series of intersecting nodes.

        Returns:
            Jaccard index as a float between 0 and 1.
        """
        jaccard_index = len(intersection) / (len(node_df1) + len(node_df2) - len(intersection))
        return jaccard_index

    def get_intersection(self, node_df1: pd.DataFrame, node_df2: pd.DataFrame) -> pd.Series:
        """
        Get the intersection of nodes between two dataframes.

        Args:
            node_df1: First DataFrame of nodes.
            node_df2: Second DataFrame of nodes.

        Returns:
            Series of nodes present in both dataframes.
        """
        intersection = pd.merge(node_df1, node_df2, on=GraphManager.NODE_COLUMNS, how='inner')[self.COLUMNS]
        return intersection

    def calculated_weighted_jaccard(self, links_lookup: dict) -> float:
        """
        Calculate weighted Jaccard index from edge weights.

        Args:
            links_lookup: Nested dict mapping target -> source -> [weight1, weight2].

        Returns:
            Weighted Jaccard index as sum(min weights) / sum(max weights).
        """
        min_weights = []
        max_weights = []
        for target, sources in links_lookup.items():
            for source, (w1, w2) in sources.items():
                min_weights.append(min(w1, w2))
                max_weights.append(max(w1, w2))
        weighted_jaccard = sum(sorted(min_weights)) / sum(sorted(max_weights)) if max_weights else 0.0
        return weighted_jaccard

    def get_links_overlap(self, graph1: GraphManager,
                          graph2: GraphManager,
                          raise_if_no_matching_tokens: bool = True) -> tuple[dict, tuple[ndarray, ndarray], dict]:
        """
        Creates a nested dictionary with all target_ids at the top level, linked source_ids at the next level
        and finally link weights as the values which are represented as lists ([weight graph 1, weight graph 2]).
        If an edge does not appear in one of the graphs, its weight is set to 0.

        Args:
            graph1: First GraphManager instance.
            graph2: Second GraphManager instance.
            raise_if_no_matching_tokens: Whether to raise if output tokens don't match.

        Returns:
            Tuple of (links_lookup dict, (intersection weights, total weights), output_node2).
        """
        output_node2 = self.get_matching_token(graph1, graph2, raise_if_no_matching_tokens)
        links1 = graph1.get_links_from_node(include_features_only=True)
        links2 = graph2.get_links_from_node(starting_node=output_node2, include_features_only=True)
        links_lookup = {}
        total_weights = {0: [], 1: []}
        intersection_weights = {0: [], 1: []}

        def add_link(target, source, weight, index):
            target, source = GraphManager.get_id_without_pos(target), GraphManager.get_id_without_pos(source)
            if target not in links_lookup:
                links_lookup[target] = {}
            links_lookup[target][source] = [weight, 0] if index == 0 else [0, weight]
            total_weights[index].append(weight)

        def add_intersecting_link(target, source, weight):
            intersection_weights[0].append(links_lookup[target][source][0])
            intersection_weights[1].append(weight)
            links_lookup[target_id][source][1] = weight
            total_weights[1].append(weight)

        for link in links1:
            add_link(link['target'], link['source'], link['weight'], 0)

        for link in links2:
            target_id = GraphManager.get_id_without_pos(link['target']),
            source_id = GraphManager.get_id_without_pos(link['source'])
            if target_id in links_lookup and source_id in links_lookup[target_id]:
                add_intersecting_link(target_id, source_id, link["weight"])
            else:
                add_link(target_id, source_id, link['weight'], 1)

        # Use sorted summation for numerical stability
        intersection = np.array([
            sum(sorted(intersection_weights[i]))
            for i in range(2)
        ])
        total = np.array([
            sum(sorted(total_weights[i]))
            for i in range(2)
        ])

        return links_lookup, (intersection, total), output_node2

    def calculate_shared_feature_metrics(self, comparison_prompts: List[str] = None,
                                         thresholds: List[int] = None,
                                         primary_threshold: int = PRIMARY_THRESHOLD,
                                         metrics: Set[SharedFeatureMetrics] = 'all') -> Dict[str, Any] | None:
        """
        Calculate metrics about feature sharing across a set of prompts.

        Args:
            comparison_prompts: List of prompt IDs to analyze. Defaults to all prompts.
            thresholds: List of percentage thresholds for shared feature counts.
            primary_threshold: Primary threshold percentage for num_shared metric.
            metrics: Set of SharedFeatureMetrics to calculate, or 'all'.

        Returns:
            Dictionary of metrics including shared counts at various thresholds, or None if no valid metrics.
        """
        metrics = self.get_metrics(metrics, SharedFeatureMetrics)
        if not metrics:
            return None
        thresholds = self.DEFAULT_THRESHOLDS if not thresholds else thresholds
        comparison_prompts = list(self.prompts.keys()) if comparison_prompts is None else comparison_prompts
        num_prompts = len(comparison_prompts)
        all_dfs = [self.dfs[p_id] for p_id in comparison_prompts]
        metric_results = {SharedFeatureMetrics.NUM_PROMPTS: num_prompts,
                          SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT: sum(len(df) for df in all_dfs) / num_prompts }

        if primary_threshold not in thresholds:
            thresholds.append(primary_threshold)

        filtered_features = {}
        counts = {}
        for threshold in thresholds:
            filtered_features[threshold] = self.get_most_freq_features_across_prompts(threshold, comparison_prompts)
            counts[threshold] = len(filtered_features[threshold] )

        metric_results[SharedFeatureMetrics.NUM_SHARED] = counts[primary_threshold]

        # Calculate how many of the shared features (primary threshold) each prompt contains
        shared_present_per_prompt = []
        for p_id in comparison_prompts:
            count = sum(1 for row in filtered_features[primary_threshold].itertuples()
                        if self.graphs[p_id].check_feature_presence(row.layer, row.feature))
            shared_present_per_prompt.append(count)

        # Store as comma-separated string for CSV compatibility
        shared_present_per_prompt = ','.join(str(c) for c in shared_present_per_prompt)
        metric_results[SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT] = shared_present_per_prompt

        metrics_at_threshold = {SharedFeatureMetrics.COUNT_AT_THRESHOLD.value.format(threshold):
                                count for threshold, count in counts.items()}

        return {metric.value: res for metric, res in metric_results.items()} | metrics_at_threshold

    def get_metrics(self, metrics: set[Metrics], metric_type: Type[Metrics]) -> Set[Metrics]:
        """
        Filter and validate a set of metrics against a metric type.

        Args:
            metrics: Set of metrics or 'all' to include all metrics of the type.
            metric_type: The Metrics enum class to filter by.

        Returns:
            Set of valid metrics matching the specified type.
        """
        metrics = {e for e in metric_type} if metrics == "all" else metrics
        metrics = {metric for metric in metrics if isinstance(metric, metric_type)}
        return metrics

    def get_most_freq_features_across_prompts(self, percent_shared: int,
                                              prompts2compare: List[str] = None,
                                              filter_layers_less_than: int = 1,
                                              filter_by_act_density=MIN_ACTIVATION_DENSITY
                                              ) -> pd.DataFrame:
        """
        Get features shared across at least the specified percentage of prompts.

        Args:
            percent_shared: Minimum percentage of prompts that must contain the feature.
            prompts2compare: List of prompt IDs to analyze. Defaults to all prompts.
            filter_layers_less_than: Exclude features from layers below this number.
            filter_by_act_density: Exclude features with activation density above this percentage.

        Returns:
            DataFrame of features meeting the frequency threshold, with ctx_freq column.
        """
        # Combine all dfs (each already de-duplicated per prompt)
        all_dfs = list(self.dfs.values())
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Get frequencies - counts how many prompts have each feature
        freq_df = GraphManager.get_frequencies(combined_df)
        threshold = (len(prompts2compare) + 1) * (percent_shared / 100)
        shared = freq_df[freq_df["ctx_freq"] >= threshold]
        return self.neuronpedia_manager.filter_features_for_subgraph(
                shared,
                filter_layers_less_than=filter_layers_less_than,
                filter_by_act_density=filter_by_act_density
                )

    @staticmethod
    def get_matching_token(graph1: GraphManager, graph2: GraphManager, raise_if_no_matching_tokens: bool) -> dict[
        str, Any]:
        """
        Find the output node in graph2 that matches graph1's top output token.

        Args:
            graph1: First GraphManager instance.
            graph2: Second GraphManager instance to search for matching token.
            raise_if_no_matching_tokens: Whether to raise if no matching token found.

        Returns:
            The matching output node from graph2, or graph2's top output if no match and not raising.
        """
        output_node1 = graph1.get_top_output_logit_node()
        output_node2 = graph2.find_output_node(output_node1)
        if not output_node2:
            if raise_if_no_matching_tokens:
                raise Exception(f"Output tokens don't match for: {output_node1['clerp']}")
            output_node2 = graph2.get_top_output_logit_node()  # just grab top
        return output_node2
