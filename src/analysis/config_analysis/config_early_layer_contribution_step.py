from typing import Dict, Any, TYPE_CHECKING

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep

if TYPE_CHECKING:
    from src.graph_analyzer import GraphAnalyzer


class ConfigEarlyLayerContributionStep(ConfigAnalyzeStep):
    """Calculates early layer contribution fraction for all prompts in config."""

    def __init__(self,
                 graph_analyzer: "GraphAnalyzer",
                 max_layer: int = 2,
                 **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            max_layer: Maximum layer to include (inclusive). Defaults to 2.
        """
        self.max_layer = max_layer
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> Dict[str, float]:
        """
        Calculates early layer contribution fraction for each prompt.

        Returns:
            Dictionary mapping prompt_id to early layer contribution fraction.
        """
        results = {}

        for prompt_id in self.graph_analyzer.prompts:
            fraction = self.graph_analyzer.get_early_layer_contribution_fraction(
                prompt_id=prompt_id,
                max_layer=self.max_layer
            )
            results[prompt_id] = fraction

        return results
