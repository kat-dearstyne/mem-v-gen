from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.metrics import FeatureSharingMetrics

# DataFrame column name
CONFIG_NAME_COL = "config_name"


class CrossConfigFeatureOverlapStep(CrossConfigAnalyzeStep):
    """
    Cross-config analysis step for feature overlap results.

    Aggregates feature sharing metrics (unique/shared fractions, weights)
    across configs and saves results to CSV.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.FEATURE_OVERLAP
    FEATURE_OVERLAP_METRICS_FILENAME = "feature-overlap-metrics.csv"

    def __init__(self, save_path: Path = None, **kwargs):
        """
        Initializes the cross-config feature overlap step.

        Args:
            save_path: Base path for saving results.
        """
        super().__init__(save_path=save_path, **kwargs)

    @property
    def metric_cols(self) -> List[str]:
        """Column names for feature sharing metrics."""
        return [m.value for m in FeatureSharingMetrics]

    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> pd.DataFrame | None:
        """
        Aggregates feature overlap results across configs and saves to CSV.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            DataFrame with combined metrics, or None if no results found.
        """
        results = self._init_results()

        has_results = False

        for config_name, step_results in config_results.items():
            overlap_results = step_results.get(self.CONFIG_RESULTS_KEY)
            if not overlap_results:
                continue

            has_results = True
            results[CONFIG_NAME_COL].append(config_name)

            for metric in FeatureSharingMetrics:
                results[metric.value].append(overlap_results.get(metric))

        if not has_results:
            return None

        df = pd.DataFrame(results)

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.save_path / self.FEATURE_OVERLAP_METRICS_FILENAME)
            print(f"Saved feature overlap metrics to: {self.save_path / self.FEATURE_OVERLAP_METRICS_FILENAME}")

        return df

    def _init_results(self) -> Dict[str, List]:
        """Initialize empty structure for feature overlap metrics."""
        return {
            CONFIG_NAME_COL: [],
            **{col: [] for col in self.metric_cols}
        }
