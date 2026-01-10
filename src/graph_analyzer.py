from typing import Callable, Dict, Tuple, Any, List, Optional, Set, Type, Union

import pandas as pd

from src.constants import MIN_ACTIVATION_DENSITY
from src.graph_manager import GraphManager, Feature
from src.metrics import (
    Metrics,
    ComparisonMetrics,
    SharedFeatureMetrics,
    FeatureSharingMetrics,
)
from src.neuronpedia_manager import NeuronpediaManager


class GraphAnalyzer:
    """Analyzer for comparing attribution graphs across multiple prompts."""

    DEFAULT_THRESHOLDS: List[int] = [50, 75, 100]
    PRIMARY_THRESHOLD: int = 50
    TOP_K_OUTPUTS: int = 5
    COLUMNS: List[str] = GraphManager.NODE_COLUMNS

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

    def get_graph_and_df(self, prompt_id: str) -> Tuple[GraphManager, pd.DataFrame]:
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
            combine_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
            verbose: bool = False,
            raise_if_no_matching_tokens: bool = False,
            metrics2run: Set[Metrics] | str = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]] | pd.DataFrame:
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
            print(f"Main prompt output: ", graph1.get_output_token_from_clerp())

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
                prompt_ids=comparison_prompts,
                metrics=metrics2run
            )
            if shared_features:
                metric_results["shared_features"] = shared_features
            return result_df, metric_results
        return result_df

    def nodes_not_in(self, main_prompt_id: str,
                     comparison_prompts: List[str] | None = None,
                     verbose: bool = False,
                     raise_if_no_matching_tokens: bool = False,
                     metrics2run: Set[Metrics] | str = None
                     ) -> Tuple[pd.DataFrame, Dict[str, Any]] | pd.DataFrame:
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
        result = self._compare_nodes(
            main_prompt_id=main_prompt_id,
            comparison_prompts=comparison_prompts,
            combine_fn=self.filter_for_unique_features,
            verbose=verbose,
            raise_if_no_matching_tokens=raise_if_no_matching_tokens,
            metrics2run=metrics2run
        )

        if metrics2run:
            result_df, metric_results = result
            _, main_df = self.get_graph_and_df(main_prompt_id)
            sharing_metrics = self.get_metrics(metrics2run, FeatureSharingMetrics)
            if sharing_metrics:
                num_main = len(main_df)
                num_unique = len(result_df)
                if FeatureSharingMetrics.NUM_MAIN in sharing_metrics:
                    metric_results[FeatureSharingMetrics.NUM_MAIN] = num_main
                if FeatureSharingMetrics.NUM_UNIQUE in sharing_metrics:
                    metric_results[FeatureSharingMetrics.NUM_UNIQUE] = num_unique
                if FeatureSharingMetrics.UNIQUE_FRAC in sharing_metrics:
                    metric_results[FeatureSharingMetrics.UNIQUE_FRAC] = num_unique / num_main if num_main > 0 else 0.0
            return result_df, metric_results

        return result

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
        result = self._compare_nodes(
            main_prompt_id=main_prompt_id,
            comparison_prompts=comparison_prompts,
            combine_fn=self.intersect_features,
            verbose=verbose,
            raise_if_no_matching_tokens=raise_if_no_matching_tokens,
            metrics2run=metrics2run
        )

        if metrics2run:
            result_df, metric_results = result
            _, main_df = self.get_graph_and_df(main_prompt_id)
            sharing_metrics = self.get_metrics(metrics2run, FeatureSharingMetrics)
            if sharing_metrics:
                num_main = len(main_df)
                num_shared = len(result_df)
                if FeatureSharingMetrics.NUM_MAIN in sharing_metrics:
                    metric_results[FeatureSharingMetrics.NUM_MAIN] = num_main
                if FeatureSharingMetrics.NUM_SHARED in sharing_metrics:
                    metric_results[FeatureSharingMetrics.NUM_SHARED] = num_shared
                if FeatureSharingMetrics.SHARED_FRAC in sharing_metrics:
                    metric_results[FeatureSharingMetrics.SHARED_FRAC] = num_shared / num_main if num_main > 0 else 0.0
            return result_df, metric_results

        return result

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
        metrics = {metric for metric in metrics if isinstance(metric, ComparisonMetrics)}
        if not metrics:
            return None
        metric_results = {}
        graph1, node_df1 = self.get_graph_and_df(prompt1_id)
        graph2, node_df2 = self.get_graph_and_df(prompt2_id)
        intersection = self.get_intersection(node_df1, node_df2)
        output_node2 = self.get_matching_token(graph1, graph2, raise_if_no_matching_tokens)
        links_lookup = self.get_links_overlap(prompt1_id, prompt2_id, starting_node2=output_node2)
        output_token = graph2.get_output_token_from_clerp(output_node2)
        if verbose:
            print(f"Prompt {prompt2_id} output: ", output_token)
            print(f"Graph URL: {graph2.url}")

        if ComparisonMetrics.WEIGHTED_JACCARD in metrics:
            metric_results[ComparisonMetrics.WEIGHTED_JACCARD] = self.calculated_weighted_jaccard(links_lookup)
        if ComparisonMetrics.JACCARD_INDEX in metrics:
            metric_results[ComparisonMetrics.JACCARD_INDEX] = self.calculated_jaccard_index(node_df1, node_df2,
                                                                                            intersection)
        if ComparisonMetrics.FRAC_FROM_INTERSECTION in metrics:
            metric_results[ComparisonMetrics.FRAC_FROM_INTERSECTION] = len(intersection) / len(node_df1)
        if ComparisonMetrics.OUTPUT_PROBABILITY in metrics:
            metric_results[ComparisonMetrics.OUTPUT_PROBABILITY] = output_node2['token_prob']
        if ComparisonMetrics.SHARED_TOKEN in metrics:
            metric_results[ComparisonMetrics.SHARED_TOKEN] = output_token
        return metric_results

    @staticmethod
    def calculated_jaccard_index(node_df1: pd.DataFrame, node_df2: pd.DataFrame,
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

    def get_combined_links_lookup(self, prompt_ids: List[str], starting_nodes: List[dict] = None) -> Dict[
        str, Dict[str, List[float]]]:
        """
        Creates a nested dictionary with all target_ids at the top level, linked source_ids at the next level
        and link weights as lists (one weight per graph, 0 if edge not present in that graph).

        Args:
            prompt_ids: List of prompt IDs to get graphs for.
            starting_nodes: Optional list of starting nodes for each graph (for filtered traversal).

        Returns:
            Nested dict mapping target_id -> source_id -> [weight per graph].
        """
        links_lookup = {}
        num_graphs = len(prompt_ids)

        for graph_idx, prompt_id in enumerate(prompt_ids):
            graph = self.graphs[prompt_id]
            starting_node = starting_nodes[graph_idx] if starting_nodes else None
            links = graph.get_links_from_node(starting_node=starting_node, include_features_only=True)
            for link in links:
                target_id = GraphManager.get_id_without_pos(link['target'])
                source_id = GraphManager.get_id_without_pos(link['source'])
                if target_id not in links_lookup:
                    links_lookup[target_id] = {}
                if source_id not in links_lookup[target_id]:
                    links_lookup[target_id][source_id] = [0.0] * num_graphs
                links_lookup[target_id][source_id][graph_idx] = link['weight']

        return links_lookup

    @staticmethod
    def calculated_weighted_jaccard(links_lookup: dict) -> float:
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

    def get_links_overlap(self, prompt1_id: str,
                          prompt2_id: str,
                          starting_node1: Dict = None,
                          starting_node2: Dict = None) -> dict:
        """
        Get link overlap between two graphs, with intersection and total weight sums.

        Args:
            prompt1_id: First prompt identifier.
            prompt2_id: Second prompt identifier.
            starting_node1: The node to start from for first graph if not the top.
            starting_node2: The node to start from for second graph if not the top.

        Returns:
            links_lookup dict.
        """
        links_lookup = self.get_combined_links_lookup([prompt1_id, prompt2_id],
                                                      starting_nodes=[starting_node1, starting_node2])

        intersection_weights = {0: [], 1: []}
        total_weights = {0: [], 1: []}
        for sources in links_lookup.values():
            for (w1, w2) in sources.values():
                if w1 > 0:
                    total_weights[0].append(w1)
                if w2 > 0:
                    total_weights[1].append(w2)
                if w1 > 0 and w2 > 0:
                    intersection_weights[0].append(w1)
                    intersection_weights[1].append(w2)

        return links_lookup

    def calculate_edge_sharing_metrics(self, prompt_ids: List[str],
                                       metrics: Set[FeatureSharingMetrics] | str = 'all') -> Dict[str, float] | None:
        """
        Calculates weighted fractions for unique-to-main and shared-among-all edges.

        Args:
            prompt_ids: List of prompt IDs where the first is the main prompt.
            metrics: Set of FeatureSharingMetrics to calculate, or 'all'.

        Returns:
            Dictionary with FeatureSharingMetrics keys containing weights and fractions, or None if no valid metrics.
        """
        metrics = self.get_metrics(metrics, FeatureSharingMetrics)
        if not metrics:
            return None

        links_lookup = self.get_combined_links_lookup(prompt_ids)
        main_total_weight = 0.0
        unique_weight = 0.0
        shared_weight = 0.0

        for target, sources in links_lookup.items():
            for source, weights in sources.items():
                main_weight = weights[0]
                other_weights = weights[1:]

                if main_weight > 0:
                    main_total_weight += main_weight

                    # Edge is unique to main if no other graph has it
                    if all(w == 0 for w in other_weights):
                        unique_weight += main_weight

                    # Edge is shared if ALL graphs have it
                    if all(w > 0 for w in other_weights):
                        shared_weight += main_weight

        unique_weighted_frac = unique_weight / main_total_weight if main_total_weight > 0 else 0.0
        shared_weighted_frac = shared_weight / main_total_weight if main_total_weight > 0 else 0.0

        results = {}
        if FeatureSharingMetrics.MAIN_TOTAL_WEIGHT in metrics:
            results[FeatureSharingMetrics.MAIN_TOTAL_WEIGHT] = main_total_weight
        if FeatureSharingMetrics.UNIQUE_WEIGHT in metrics:
            results[FeatureSharingMetrics.UNIQUE_WEIGHT] = unique_weight
        if FeatureSharingMetrics.SHARED_WEIGHT in metrics:
            results[FeatureSharingMetrics.SHARED_WEIGHT] = shared_weight
        if FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC in metrics:
            results[FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC] = unique_weighted_frac
        if FeatureSharingMetrics.SHARED_WEIGHTED_FRAC in metrics:
            results[FeatureSharingMetrics.SHARED_WEIGHTED_FRAC] = shared_weighted_frac

        return results

    def find_features(self, prompt_id: str, features: pd.DataFrame, create_subgraph: bool = True) -> pd.DataFrame:
        """
        Find which features from a given set are present in a prompt's graph.

        Args:
            prompt_id: The prompt identifier to search in.
            features: DataFrame with layer and feature columns to search for.
            create_subgraph: Whether to create a subgraph from found features.

        Returns:
            DataFrame of features that were found in the graph.
        """
        graph, _ = self.get_graph_and_df(prompt_id)
        features_found = [(row.layer, row.feature) for row in features.itertuples()
                          if graph.check_feature_presence(row.layer, row.feature)]
        selected = pd.DataFrame({
            "feature": [f[-1] for f in features_found],
            "layer": [f[0] for f in features_found],
        })
        if len(selected) > 0 and create_subgraph:
            self.neuronpedia_manager.create_subgraph_from_selected_features(selected, graph,
                                                                            list_name=f"Common features.")
        return selected

    def calculate_shared_feature_metrics(self, prompt_ids: List[str] = None,
                                         thresholds: List[int] = None,
                                         primary_threshold: int = PRIMARY_THRESHOLD,
                                         metrics: Set[SharedFeatureMetrics | Metrics] = 'all') -> Dict[str, Any] | None:
        """
        Calculate metrics about feature sharing across a set of prompts.

        Args:
            prompt_ids: List of prompt IDs to analyze. Defaults to all prompts.
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
        prompt_ids = list(self.prompts.keys()) if prompt_ids is None else prompt_ids
        num_prompts = len(prompt_ids)
        all_dfs = [self.dfs[p_id] for p_id in prompt_ids]
        metric_results = {SharedFeatureMetrics.NUM_PROMPTS: num_prompts,
                          SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT: sum(len(df) for df in all_dfs) / num_prompts}

        if primary_threshold not in thresholds:
            thresholds.append(primary_threshold)

        filtered_features = {}
        counts = {}
        for threshold in thresholds:
            filtered_features[threshold] = self.get_most_freq_features_across_prompts(threshold, prompt_ids)
            counts[threshold] = len(filtered_features[threshold])

        metric_results[SharedFeatureMetrics.NUM_SHARED] = counts[primary_threshold]

        # Calculate how many of the shared features (primary threshold) each prompt contains
        shared_present_per_prompt = []
        for p_id in prompt_ids:
            found_features = self.find_features(prompt_id=p_id, features=filtered_features[primary_threshold])
            count = len(found_features)
            shared_present_per_prompt.append(count)

        # Store as comma-separated string for CSV compatibility
        shared_present_per_prompt = ','.join(str(c) for c in shared_present_per_prompt)
        metric_results[SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT] = shared_present_per_prompt

        # COUNT_AT_THRESHOLD uses string keys since threshold values are dynamic
        metrics_at_threshold = {SharedFeatureMetrics.COUNT_AT_THRESHOLD.value.format(threshold):
                                    count for threshold, count in counts.items()}

        return metric_results | metrics_at_threshold

    @staticmethod
    def get_metrics(metrics: set[Metrics], metric_type: Type[Metrics]) -> Set[Metrics]:
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

    def compare_token_subgraphs(self, prompt_id: str, token_of_interest: str,
                                top_k_tokens: int = TOP_K_OUTPUTS) -> List[Feature]:
        """
        Compares subgraphs for different output tokens and returns features unique to the token of interest.

        Args:
            prompt_id: The prompt identifier to analyze.
            token_of_interest: The output token to find unique features for.
            top_k_tokens: Number of top output tokens to compare against.

        Returns:
            List of Feature namedtuples unique to the token of interest.
        """
        graph = self.neuronpedia_manager.create_or_load_graph(prompt=prompt_id)
        output_token_to_linked_features = graph.get_features_linked_to_tokens(top_k_tokens)
        unique_features = self.get_unique_features_for_token(token_of_interest,
                                                             output_token_to_linked_features)
        return unique_features

    @staticmethod
    def get_unique_features_for_token(token_of_interest: str,
                                      output_token_to_linked_features: Dict[str, Set[str]]) -> List[Feature]:
        """
        Gets features unique to a specific token (not linked to any other tokens).

        Args:
            token_of_interest: The token to find unique features for.
            output_token_to_linked_features: Dict mapping tokens to sets of linked node ids.

        Returns:
            List of Feature namedtuples unique to the specified token.
        """
        # all features except those linked to token of interest
        other_features = set()
        for token, node_ids in output_token_to_linked_features.items():
            if token != token_of_interest:
                other_features.update(node_ids)
        assert token_of_interest in output_token_to_linked_features, f"Token {token_of_interest} not found"
        unique_node_ids = output_token_to_linked_features[token_of_interest].difference(other_features)
        unique_features = [GraphManager.get_feature_from_node_id(node_id, deliminator="_") for node_id in
                           unique_node_ids]
        return unique_features

    def get_most_frequent_features(self, prompt_id: str, frequency_count: int = 1) -> pd.DataFrame:
        """
        Get features that appear more than the specified frequency count in a graph.

        Args:
            prompt_id: The prompt identifier to analyze.
            frequency_count: Minimum frequency threshold (exclusive).

        Returns:
            DataFrame of features with ctx_freq greater than frequency_count.
        """
        graph, _ = self.get_graph_and_df(prompt_id)
        main_node_df_full = graph.create_node_df(exclude_embeddings=True, exclude_errors=True,
                                                 exclude_logits=True)
        freq_df = graph.get_frequencies_from_graph(main_node_df_full)
        frequent_features = freq_df[freq_df['ctx_freq'] > frequency_count][GraphManager.NODE_COLUMNS]
        return frequent_features

    @staticmethod
    def get_matching_token(graph1: GraphManager, graph2: GraphManager,
                           raise_if_no_matching_tokens: bool) -> dict[str, Any]:
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
                raise Exception(f"Output tokens don't match for: {graph1.get_output_token_from_clerp()}")
            output_node2 = graph2.get_top_output_logit_node()  # just grab top
        return output_node2

    def get_early_layer_contribution_fraction(self,
                                               prompt_id: str,
                                               max_layer: int = 2,
                                               starting_node: Dict = None) -> float:
        """
        Get the fraction of total contribution from features at early layers.

        Args:
            prompt_id: The prompt identifier to analyze.
            max_layer: Maximum layer to include (inclusive). Defaults to 2.
            starting_node: Node to start traversal from. Defaults to top output logit.

        Returns:
            Fraction of total weight from features at layer <= max_layer.
        """
        graph, _ = self.get_graph_and_df(prompt_id)
        node_dict = graph.node_dict

        links = graph.get_links_from_node(
            starting_node=starting_node,
            include_features_only=True
        )

        total_weight = 0.0
        early_layer_weight = 0.0

        for link in links:
            weight = link['weight']
            total_weight += weight

            source_node = node_dict.get(link['source'])
            if source_node and float(source_node.get('layer', 'inf')) <= max_layer:
                early_layer_weight += weight

        return early_layer_weight / total_weight if total_weight > 0 else 0.0
