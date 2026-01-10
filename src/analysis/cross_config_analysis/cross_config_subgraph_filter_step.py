from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.graph_analyzer import GraphAnalyzer
from src.metrics import ComparisonMetrics, SharedFeatureMetrics


# Result dictionary keys
DIFF_KEY = "diff"
SIM_KEY = "sim"
SHARED_FEATURES_KEY = "shared_features"

# DataFrame column names
CONFIG_NAME_COL = "config_name"
PROMPT_TYPE_COL = "prompt_type"


class CrossConfigSubgraphFilterStep(CrossConfigAnalyzeStep):
    """
    Cross-config analysis step for subgraph filter results.

    Aggregates intersection metrics and shared feature metrics across configs,
    saves results to CSV files.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.SUBGRAPH_FILTER
    OVERLAP_ANALYSIS_FILENAME = "overlap-analysis.csv"
    SHARED_FEATURE_METRICS_FILENAME = "shared-feature-metrics.csv"

    def __init__(self, save_path: Path = None,
                 thresholds: List[int] = None, **kwargs):
        """
        Initializes the cross-config subgraph filter step.

        Args:
            save_path: Base path for saving results.
            thresholds: List of threshold values for count_at metrics.
        """
        super().__init__(save_path=save_path, **kwargs)
        self.thresholds = thresholds or GraphAnalyzer.DEFAULT_THRESHOLDS

    @property
    def intersection_metric_cols(self) -> List[str]:
        """Column names for intersection metrics."""
        return [m.value for m in ComparisonMetrics]

    @property
    def shared_feature_metric_cols(self) -> List[str]:
        """Column names for shared feature metrics."""
        base_cols = [
            SharedFeatureMetrics.NUM_SHARED.value,
            SharedFeatureMetrics.NUM_PROMPTS.value,
            SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT.value,
            SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT.value,
        ]
        threshold_cols = [
            SharedFeatureMetrics.COUNT_AT_THRESHOLD.value.format(t)
            for t in self.thresholds
        ]
        return base_cols + threshold_cols

    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Dict[str, pd.DataFrame] | None:
        """
        Aggregates subgraph filter results across configs and saves to CSV.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            Dictionary with 'intersection_metrics' and 'shared_features' DataFrames,
            or None if no results found.
        """
        intersection_results = self._init_intersection_results()
        shared_feature_results = self._init_shared_feature_results()

        has_results = False

        for config_name, step_results in config_results.items():
            subgraph_results = step_results.get(self.CONFIG_RESULTS_KEY)
            if not subgraph_results:
                continue

            has_results = True

            # Process both diff and sim results
            for result_key in [DIFF_KEY, SIM_KEY]:
                step_data = subgraph_results.get(result_key)
                if not step_data:
                    continue

                self._extract_intersection_metrics(
                    intersection_results, step_data, config_name
                )
                self._extract_shared_features(
                    shared_feature_results, step_data, config_name
                )

        if not has_results:
            return None

        result = {}

        if intersection_results[CONFIG_NAME_COL]:
            intersection_df = pd.DataFrame(intersection_results)
            result["intersection_metrics"] = intersection_df

            if self.save_path:
                self.save_path.mkdir(parents=True, exist_ok=True)
                intersection_df.to_csv(self.save_path / self.OVERLAP_ANALYSIS_FILENAME)
                print(f"Saved intersection metrics to: {self.save_path / self.OVERLAP_ANALYSIS_FILENAME}")

        if shared_feature_results[CONFIG_NAME_COL]:
            shared_feature_df = pd.DataFrame(shared_feature_results)
            result[SHARED_FEATURES_KEY] = shared_feature_df

            if self.save_path:
                shared_feature_df.to_csv(self.save_path / self.SHARED_FEATURE_METRICS_FILENAME)
                print(f"Saved shared feature metrics to: {self.save_path / self.SHARED_FEATURE_METRICS_FILENAME}")

        return result if result else None

    def _init_intersection_results(self) -> Dict[str, List]:
        """Initialize empty structure for intersection metrics."""
        return {
            CONFIG_NAME_COL: [],
            PROMPT_TYPE_COL: [],
            **{col: [] for col in self.intersection_metric_cols}
        }

    def _init_shared_feature_results(self) -> Dict[str, List]:
        """Initialize empty structure for shared feature metrics."""
        return {
            CONFIG_NAME_COL: [],
            **{col: [] for col in self.shared_feature_metric_cols}
        }

    def _extract_intersection_metrics(self, results: Dict[str, List],
                                       step_data: Dict[str, Any],
                                       config_name: str) -> None:
        """
        Extract intersection metrics from step data.

        Args:
            results: Results dictionary to append to (modified in place).
            step_data: Data from a single result key (diff or sim).
            config_name: Name of the config being processed.
        """
        for prompt, metrics_dict in step_data.items():
            if prompt == SHARED_FEATURES_KEY:
                continue

            results[CONFIG_NAME_COL].append(config_name)
            results[PROMPT_TYPE_COL].append(prompt)

            for metric in ComparisonMetrics:
                results[metric.value].append(metrics_dict.get(metric))

    def _extract_shared_features(self, results: Dict[str, List],
                                  step_data: Dict[str, Any],
                                  config_name: str) -> None:
        """
        Extract shared feature metrics from step data.

        Args:
            results: Results dictionary to append to (modified in place).
            step_data: Data from a single result key (diff or sim).
            config_name: Name of the config being processed.
        """
        shared_features = step_data.get(SHARED_FEATURES_KEY)
        if not shared_features:
            return

        results[CONFIG_NAME_COL].append(config_name)

        # Extract enum-keyed metrics
        for metric in [SharedFeatureMetrics.NUM_SHARED, SharedFeatureMetrics.NUM_PROMPTS,
                       SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT,
                       SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT]:
            results[metric.value].append(shared_features.get(metric))

        # Extract threshold-based metrics (string keys)
        for threshold in self.thresholds:
            key = SharedFeatureMetrics.COUNT_AT_THRESHOLD.value.format(threshold)
            results[key].append(shared_features.get(key))
