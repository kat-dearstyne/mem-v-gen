from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CONFIG_NAME_COL, PROMPT_TYPE_COL
)
from src.metrics import EarlyLayerMetrics

# Column names derived from enum
EARLY_LAYER_FRACTION_COL = EarlyLayerMetrics.EARLY_LAYER_FRACTION.value
MAX_LAYER_COL = EarlyLayerMetrics.MAX_LAYER.value


class CrossConfigEarlyLayerContributionStep(CrossConfigAnalyzeStep):
    """
    Cross-config analysis step for early layer contribution results.

    Aggregates early layer contribution fractions across configs and prompt types,
    saving results to CSV.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION
    EARLY_LAYER_CONTRIBUTION_FILENAME = "early-layer-contribution.csv"

    def __init__(self, save_path: Path = None, **kwargs):
        """
        Initializes the cross-config early layer contribution step.

        Args:
            save_path: Base path for saving results.
        """
        super().__init__(save_path=save_path, **kwargs)

    @property
    def metric_cols(self) -> List[str]:
        """Column names for early layer metrics."""
        return [m.value for m in EarlyLayerMetrics]

    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> pd.DataFrame | None:
        """
        Aggregates early layer contribution results across configs and saves to CSV.

        Args:
            config_results: Dictionary mapping config names to their per-step results.
                Each config's results map prompt_type to dict of max_layer -> fraction.

        Returns:
            DataFrame with config_name, prompt_type, max_layer, and early_layer_fraction columns,
            or None if no results found.
        """
        rows = []

        for config_name, step_results in config_results.items():
            contribution_results = step_results.get(self.CONFIG_RESULTS_KEY)
            if not contribution_results:
                continue

            for prompt_type, layer_fractions in contribution_results.items():
                for max_layer, fraction in layer_fractions.items():
                    rows.append({
                        CONFIG_NAME_COL: config_name,
                        PROMPT_TYPE_COL: prompt_type,
                        EarlyLayerMetrics.MAX_LAYER.value: max_layer,
                        EarlyLayerMetrics.EARLY_LAYER_FRACTION.value: fraction
                    })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.save_path / self.EARLY_LAYER_CONTRIBUTION_FILENAME, index=False)
            print(f"Saved early layer contribution to: {self.save_path / self.EARLY_LAYER_CONTRIBUTION_FILENAME}")

        return df
