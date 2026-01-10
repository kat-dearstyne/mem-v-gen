from typing import TYPE_CHECKING

import pandas as pd

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.utils import get_method_kwargs

if TYPE_CHECKING:
    from src.graph_analyzer import GraphAnalyzer


class ConfigTokenSubgraphStep(ConfigAnalyzeStep):
    """Compares subgraphs for different output tokens to find unique features."""

    def __init__(self,
                 graph_analyzer: "GraphAnalyzer",
                 prompt_id: str,
                 token_of_interest: str,
                 create_subgraph: bool = True,
                 **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            prompt_id: The prompt ID to analyze.
            token_of_interest: The output token to find unique features for.
            **kwargs: Additional args passed to compare_token_subgraphs.
        """
        self.prompt_id = prompt_id
        self.token_of_interest = token_of_interest
        self.create_subgraph = create_subgraph
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> pd.DataFrame:
        """
        Runs token subgraph comparison.

        Returns:
            DataFrame of unique features for the token of interest.
        """
        method_kwargs = get_method_kwargs(self.graph_analyzer.compare_token_subgraphs, self.kwargs)
        unique_features = self.graph_analyzer.compare_token_subgraphs(
            prompt_id=self.prompt_id,
            token_of_interest=self.token_of_interest,
            **method_kwargs
        )
        # Create df for features
        node_dict = [{"layer": feature.layer, "feature": feature.feature} for feature in unique_features if
                     int(feature.layer) > 1]  # list of all unique features at layers higher than 1
        node_df = pd.DataFrame(node_dict)

        if self.create_subgraph:
            graph, _ = self.graph_analyzer.get_graph_and_df(self.prompt_id)
            self.graph_analyzer.neuronpedia_manager.create_subgraph_from_selected_features(
                node_df, graph,

                f"Unique Features for {self.token_of_interest}",
                         include_output_node=False)

        return node_df
