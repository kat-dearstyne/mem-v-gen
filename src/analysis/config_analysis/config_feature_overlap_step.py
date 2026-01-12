from datetime import datetime
from typing import Dict, List, Set, Any, Optional, TYPE_CHECKING

import pandas as pd

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.graph_analyzer import GraphAnalyzer
from src.graph_manager import GraphManager
from src.metrics import FeatureSharingMetrics


class ConfigFeatureOverlapStep(ConfigAnalyzeStep):
    """Compares main prompt against comparison prompts to find unique and shared features."""

    def __init__(self,
                 graph_analyzer: GraphAnalyzer,
                 main_prompt_id: str,
                 comparison_prompt_ids: List[str],
                 debug: bool = False,
                 create_subgraphs: bool = True,
                 filter_by_act_density: Optional[int] = None,
                 **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            main_prompt_id: The main prompt ID to compare from.
            comparison_prompt_ids: List of prompt IDs to compare against.
            debug: Whether to print debug output.
            create_subgraphs: Whether to create subgraphs on Neuronpedia.
            filter_by_act_density: Filter threshold for activation density.
        """
        self.main_prompt_id = main_prompt_id
        self.comparison_prompt_ids = comparison_prompt_ids
        self.debug = debug
        self.create_subgraphs = create_subgraphs
        self.filter_by_act_density = filter_by_act_density
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> Dict[FeatureSharingMetrics, Any]:
        """
        Runs feature overlap analysis comparing main prompt to comparison prompts.

        Returns:
            Dictionary with FeatureSharingMetrics keys containing sharing metrics.
        """
        main_graph, _ = self.graph_analyzer.get_graph_and_df(self.main_prompt_id)

        if self.debug:
            print(f"Main prompt output: {main_graph.get_output_token_from_clerp()}")
        print(f"Graph URL: {main_graph.url}")

        unique_metrics = {FeatureSharingMetrics.UNIQUE_FRAC, FeatureSharingMetrics.NUM_UNIQUE,
                          FeatureSharingMetrics.NUM_MAIN}
        shared_metrics = {FeatureSharingMetrics.SHARED_FRAC, FeatureSharingMetrics.NUM_SHARED}

        unique_features, unique_results = self.graph_analyzer.nodes_not_in(
            main_prompt_id=self.main_prompt_id,
            comparison_prompts=self.comparison_prompt_ids,
            raise_if_no_matching_tokens=False,
            verbose=False,
            metrics2run=unique_metrics
        )

        shared_features, shared_results = self.graph_analyzer.nodes_in(
            main_prompt_id=self.main_prompt_id,
            comparison_prompts=self.comparison_prompt_ids,
            verbose=False,
            metrics2run=shared_metrics
        )

        edge_metrics = self.graph_analyzer.calculate_edge_sharing_metrics(
            [self.main_prompt_id] + self.comparison_prompt_ids
        )

        if self.create_subgraphs:
            self._create_subgraphs(main_graph, unique_features, shared_features)

        return unique_results | shared_results | edge_metrics

    def _create_subgraphs(self, main_graph, unique_features: pd.DataFrame,
                          shared_features: pd.DataFrame) -> None:
        """
        Creates subgraphs for unique, shared, and frequently shared features.

        Args:
            main_graph: The main graph to create subgraphs from.
            unique_features: DataFrame of unique features.
            shared_features: DataFrame of shared features.
        """
        neuronpedia_manager = self.graph_analyzer.neuronpedia_manager
        timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")

        unique_filtered = neuronpedia_manager.filter_features_for_subgraph(unique_features, main_graph)
        if len(unique_filtered) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                unique_filtered, main_graph,
                list_name=f"Unique Features ({timestamp_str})"
            )

        shared_filtered = neuronpedia_manager.filter_features_for_subgraph(
            shared_features, main_graph,
            filter_by_act_density=self.filter_by_act_density
        )
        if len(shared_filtered) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                shared_filtered, main_graph,
                list_name=f"Shared Features ({timestamp_str})"
            )

        frequent_features = self.graph_analyzer.get_most_frequent_features(self.main_prompt_id)
        shared_frequent = pd.merge(shared_filtered, frequent_features, on=GraphManager.NODE_COLUMNS, how='inner')
        if len(shared_frequent) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                shared_frequent, main_graph,
                list_name=f"Shared Features Frequent Fliers ({timestamp_str})"
            )
