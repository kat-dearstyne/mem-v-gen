from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.metrics import ComparisonMetrics
from src.visualizations import (
    plot_metric_by_condition,
    plot_metric_heatmap,
    plot_metric_boxplot,
    plot_metric_line,
    plot_metric_vs_probability_scatter,
    plot_metric_vs_probability_combined,
    plot_correlation_heatmap,
)

# Result dictionary keys
INTERSECTION_METRICS_KEY = "intersection_metrics"

# Output filenames
COMBINED_RESULTS_FILENAME = "combined_results.csv"


# Metrics to visualize with standard suite (bar, heatmap, boxplot, line)
VISUALIZATION_METRICS = [
    ComparisonMetrics.JACCARD_INDEX,
    ComparisonMetrics.WEIGHTED_JACCARD,
    ComparisonMetrics.OUTPUT_PROBABILITY,
]


def metric_to_title(metric: ComparisonMetrics) -> str:
    """
    Converts metric enum to display title.

    Args:
        metric: The comparison metric enum value.

    Returns:
        Title-cased string with underscores replaced by spaces
        (e.g., 'jaccard_index' -> 'Jaccard Index').
    """
    return metric.value.replace('_', ' ').title()


class CrossConditionOverlapVisualizationStep(CrossConditionAnalyzeStep):
    """
    Cross-condition step for overlap analysis visualizations.

    Combines intersection metrics from multiple conditions and generates
    Jaccard index, weighted Jaccard, and probability visualizations.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.SUBGRAPH_FILTER
    RESULTS_SUB_KEY = INTERSECTION_METRICS_KEY

    def __init__(self, save_path: Path = None,
                 condition_order: Optional[List[str]] = None,
                 config_order: Optional[List[str]] = None,
                 extra_series: Optional[dict] = None,
                 **kwargs):
        super().__init__(save_path=save_path, condition_order=condition_order,
                         config_order=config_order, **kwargs)
        self.extra_series = extra_series

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[pd.DataFrame]:
        """
        Combines intersection metrics and generates visualizations.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Combined DataFrame of intersection metrics, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, config_order = self.get_ordering(combined_df)
        self._generate_visualizations(combined_df, condition_order, config_order)

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(self.save_path / COMBINED_RESULTS_FILENAME, index=False)

        return combined_df

    def _generate_visualizations(self, df: pd.DataFrame,
                                  condition_order: List[str],
                                  config_order: List[str]) -> None:
        """
        Generates all overlap visualizations.

        Args:
            df: Combined intersection metrics DataFrame.
            condition_order: Order for conditions in plots.
            config_order: Order for configs in plots.
        """
        for metric in VISUALIZATION_METRICS:
            if metric.value in df.columns:
                self._plot_metric_suite(df, metric, condition_order, config_order)

        self._plot_cross_metric_visualizations(df, condition_order)

    def _plot_metric_suite(self, df: pd.DataFrame,
                           metric: ComparisonMetrics,
                           condition_order: List[str],
                           config_order: List[str]) -> None:
        """
        Generates bar, heatmap, boxplot, and line plots for a metric.

        Args:
            df: DataFrame with metric data.
            metric: The metric to visualize.
            condition_order: Order for conditions in plots.
            config_order: Order for configs in plots.
        """
        save_path = self.save_path
        col = metric.value
        title = metric_to_title(metric)

        # Bar chart
        plot_metric_by_condition(
            df, metric_col=col,
            title=f'{title} by Condition',
            ylabel=title,
            condition_order=condition_order, config_order=config_order,
            save_path=save_path / f"{col}_bar.png" if save_path else None
        )

        # Heatmap
        plot_metric_heatmap(
            df, metric_col=col,
            title=f'{title} Heatmap',
            cbar_label=title,
            condition_order=condition_order, config_order=config_order,
            save_path=save_path / f"{col}_heatmap.png" if save_path else None
        )

        # Boxplot
        plot_metric_boxplot(
            df, metric_col=col,
            title=f'{title} Distribution by Condition',
            ylabel=title,
            condition_order=condition_order,
            save_path=save_path / f"{col}_boxplot.png" if save_path else None
        )

        # Line plot (with optional extra series for jaccard)
        extra = self.extra_series if metric == ComparisonMetrics.JACCARD_INDEX else None
        plot_metric_line(
            df, metric_col=col,
            title=f'{title} by Config',
            ylabel=title,
            condition_order=condition_order, config_order=config_order,
            extra_series=extra,
            save_path=save_path / f"{col}_line.png" if save_path else None
        )

    def _plot_cross_metric_visualizations(self, df: pd.DataFrame,
                                           condition_order: List[str]) -> None:
        """
        Generates visualizations comparing multiple metrics.

        Creates scatter plots showing relationships between Jaccard metrics
        and output probability, plus correlation heatmaps.

        Args:
            df: DataFrame with metric data.
            condition_order: Order for conditions in plots.
        """
        save_path = self.save_path
        jaccard_col = ComparisonMetrics.JACCARD_INDEX.value
        weighted_col = ComparisonMetrics.WEIGHTED_JACCARD.value
        prob_col = ComparisonMetrics.OUTPUT_PROBABILITY.value

        has_probability = prob_col in df.columns
        has_jaccard = jaccard_col in df.columns
        has_weighted = weighted_col in df.columns

        if not has_probability or not has_jaccard:
            return

        jaccard_title = metric_to_title(ComparisonMetrics.JACCARD_INDEX)
        plot_metric_vs_probability_scatter(
            df, metric_col=jaccard_col,
            title=f'{jaccard_title} vs Output Probability',
            xlabel=jaccard_title,
            condition_order=condition_order,
            save_path=save_path / f"{jaccard_col}_vs_probability.png" if save_path else None
        )

        if has_weighted:
            weighted_title = metric_to_title(ComparisonMetrics.WEIGHTED_JACCARD)
            plot_metric_vs_probability_scatter(
                df, metric_col=weighted_col,
                title=f'{weighted_title} vs Output Probability',
                xlabel=weighted_title,
                condition_order=condition_order,
                save_path=save_path / f"{weighted_col}_vs_probability.png" if save_path else None
            )

            plot_metric_vs_probability_combined(
                df, condition_order=condition_order,
                save_path=save_path / "jaccard_metrics_vs_probability.png" if save_path else None
            )

        plot_correlation_heatmap(
            df, condition_order=condition_order,
            save_path=save_path / "metric_correlations.png" if save_path else None
        )
