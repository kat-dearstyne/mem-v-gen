from typing import Dict, List, Optional, TYPE_CHECKING

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.graph_analyzer import GraphAnalyzer


class ConfigEarlyLayerContributionStep(ConfigAnalyzeStep):
    """Calculates early layer contribution fraction for all prompts in config."""

    def __init__(self,
                 graph_analyzer: GraphAnalyzer,
                 max_layer: Optional[int] = None,
                 **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            max_layer: Maximum layer to include (inclusive). Defaults to 2.
                If None, calculates for all layers from 0 to last layer.
        """
        self.max_layer = max_layer
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> Dict[str, Dict[int, float]]:
        """
        Calculates early layer contribution fraction for each prompt.

        Returns:
            Dictionary mapping prompt_id to dict of max_layer -> fraction.
        """
        results = {}

        for prompt_id in self.graph_analyzer.prompts:
            graph, _ = self.graph_analyzer.get_graph_and_df(prompt_id)

            if self.max_layer is not None:
                max_layers = [self.max_layer]
            else:
                last_layer = graph.get_last_layers()
                max_layers = list(range(last_layer))

            layer_fractions = {}
            for layer in max_layers:
                fraction = self.graph_analyzer.get_early_layer_contribution_fraction(
                    prompt_id=prompt_id,
                    max_layer=layer
                )
                layer_fractions[layer] = fraction

            results[prompt_id] = layer_fractions

        return results
