from typing import Dict, List, Set, Any, TYPE_CHECKING

import pandas as pd

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.constants import MIN_ACTIVATION_DENSITY
from src.graph_analyzer import GraphAnalyzer
from src.graph_manager import GraphManager
from src.metrics import ComparisonMetrics
from src.utils import get_method_kwargs


class ConfigSubgraphFilterStep(ConfigAnalyzeStep):
    """Filters subgraph to features shared with some prompts and unique from others."""

    def __init__(self,
                 graph_analyzer: GraphAnalyzer,
                 main_prompt_id: str,
                 prompts_with_shared_features: List[str] = None,
                 prompts_with_unique_features: List[str] = None,
                 metrics2run: Set[ComparisonMetrics] = None,
                 create_subgraph: bool = False,
                 **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            main_prompt_id: The main prompt ID to compare from.
            prompts_with_shared_features: Prompt IDs to find shared features with.
            prompts_with_unique_features: Prompt IDs to find unique features against.
            metrics2run: Set of ComparisonMetrics to calculate.
            **kwargs: Additional args passed to compare_prompt_groups.
        """
        self.main_prompt_id = main_prompt_id
        self.prompts_with_shared_features = prompts_with_shared_features
        self.prompts_with_unique_features = prompts_with_unique_features
        self.metrics2run = metrics2run
        self.create_subgraph = create_subgraph
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> Dict[str, Any]:
        """
        Runs subgraph filtering based on prompt feature criteria.

        Returns:
            Dictionary with 'diff' and 'sim' results.
        """
        main_graph, _ = self.graph_analyzer.get_graph_and_df(self.main_prompt_id)
        results = {}
        unique_features, overlapping_features = None, None

        if self.prompts_with_unique_features:
            unique_features, results['diff'] = self.graph_analyzer.nodes_not_in(self.main_prompt_id,
                                                                                self.prompts_with_unique_features,
                                                                                metrics2run=self.metrics2run)
        if self.prompts_with_shared_features:
            overlapping_features, results['sim'] = self.graph_analyzer.nodes_in(self.main_prompt_id,
                                                                                self.prompts_with_shared_features,
                                                                                metrics2run=self.metrics2run)

        if unique_features is not None and overlapping_features is not None:
            features_of_interest = pd.merge(overlapping_features, unique_features, how='inner',
                                            on=GraphManager.NODE_COLUMNS)
        else:
            features_of_interest = unique_features if unique_features is not None else overlapping_features

        if self.create_subgraph:
            features_of_interest = self.graph_analyzer.neuronpedia_manager.filter_features_for_subgraph(
                features_of_interest, main_graph, filter_by_act_density=MIN_ACTIVATION_DENSITY
            )

            self.graph_analyzer.neuronpedia_manager.create_subgraph_from_selected_features(
                features_of_interest, main_graph, list_name=f"Features of Interest"
            )

        return results
