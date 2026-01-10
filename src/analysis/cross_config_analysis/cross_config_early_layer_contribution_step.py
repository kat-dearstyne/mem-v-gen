from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep

CONFIG_NAME_COL = "config_name"
PROMPT_TYPE_COL = "prompt_type"
EARLY_LAYER_FRACTION_COL = "early_layer_fraction"


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

    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> pd.DataFrame | None:
        """
        Aggregates early layer contribution results across configs and saves to CSV.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            DataFrame with config_name, prompt_type, and early_layer_fraction columns,
            or None if no results found.
        """
        rows = []

        for config_name, step_results in config_results.items():
            contribution_results = step_results.get(self.CONFIG_RESULTS_KEY)
            if not contribution_results:
                continue

            for prompt_type, fraction in contribution_results.items():
                rows.append({
                    CONFIG_NAME_COL: config_name,
                    PROMPT_TYPE_COL: prompt_type,
                    EARLY_LAYER_FRACTION_COL: fraction
                })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.save_path / self.EARLY_LAYER_CONTRIBUTION_FILENAME)
            print(f"Saved early layer contribution to: {self.save_path / self.EARLY_LAYER_CONTRIBUTION_FILENAME}")

        return df
