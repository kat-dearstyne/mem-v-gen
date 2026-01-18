from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.config_analysis.config_error_ranking_step import ConfigErrorRankingStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.metrics import PooledStatsMetrics, ConditionStatsMetrics, RawScoreMetrics, ErrorRankingMetrics
from src.utils import append_to_dict_list, get_conditions_from_label, create_label_from_conditions
from src.visualizations import boxplot_metric_family, plot_delta_distribution


class CrossConfigErrorRankingStep(CrossConfigAnalyzeStep):

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.ERROR_RANKING

    def __init__(self, save_path: Path = None, **kwargs):
        """
        Initializes the cross-config error ranking step.

        Args:
            save_path: Base path for saving results
        """
        super().__init__(save_path=save_path, **kwargs)

    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Dict:
        """
        Runs the step logic.

        Args:
            config_results: Results for all configs.

        Returns:
            Dictionary of results.
        """
        return self.analyze_conditions(config_results)

    def analyze_conditions(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Dict[str, Any] | None:
        """
        Performs statistical analysis and visualization of error ranking metrics.

        Args:
            config_results: Dictionary mapping sample_id -> condition_name -> results_dict

        Returns:
            Dictionary mapping condition_name -> (stats, deltas) tuple
        """

        sample_ids = [config_name for config_name, config_results in config_results.items()
                      if self.CONFIG_RESULTS_KEY in config_results]
        if not sample_ids:
            return None

        # Get all condition names from the first sample
        first_sample = next(iter(config_results.values()))[self.CONFIG_RESULTS_KEY]
        condition_names = list(first_sample.keys())

        all_results = {}
        all_deltas = {}

        for condition_name in condition_names:
            condition_results = [config_results[sample_id][self.CONFIG_RESULTS_KEY][condition_name] for sample_id in sample_ids]

            deltas = self.extract_metric_deltas(condition_results)
            stats = self.condition_level_stats(deltas)
            g1_values, g2_values = self.extract_raw_values(condition_results)

            if self.save_path:
                condition_save_path = self.save_path / condition_name
                self._save_results(
                    condition_save_path, stats, deltas, condition_name,
                    g1_values=g1_values, g2_values=g2_values, sample_ids=sample_ids
                )

            all_results[condition_name] = (stats, deltas)
            all_deltas[condition_name] = deltas

        # Pooled analysis across all conditions
        condition_pairs = [get_conditions_from_label(condition) for condition in condition_names]
        assert all(x[0] == condition_pairs[0][0] for x in condition_pairs), \
            "Cannot pool results because condition1 is not the same across pairs"
        main_condition = condition_pairs[0][0]
        other_conditions = "  and ".join([pair[1] for pair in condition_pairs])
        pooled_stats, pooled_deltas = self.pooled_condition_stats(all_deltas)

        if self.save_path:
            pooled_save_path = self.save_path / "pooled"
            self._save_results(pooled_save_path, pooled_stats, pooled_deltas,
                               create_label_from_conditions(main_condition, other_conditions), is_pooled=True)

        all_results["pooled"] = (pooled_stats, pooled_deltas)

        return all_results

    def raw_scores_to_csv(self, g1_values: Dict[ErrorRankingMetrics, Any],
                          g2_values: Dict[ErrorRankingMetrics, Any],
                          deltas: Dict[ErrorRankingMetrics, Any], filepath: Path,
                          sample_ids: List[str] = None):
        """
        Saves individual sample scores to a CSV file.

        Args:
            g1_values: Metric values for graph 1.
            g2_values: Metric values for graph 2.
            deltas: Delta values between graphs.
            filepath: Path to save CSV.
            sample_ids: Optional list of sample identifiers.

        Returns:
            DataFrame containing the saved data.
        """
        rows = []

        def add_scores_to_row(metric: ErrorRankingMetrics, k, g1_list):
            g2_list = g2_values[metric] if k is None else g2_values[metric][k]
            d_list = deltas[metric] if k is None else deltas[metric][k]
            for i, (g1, g2, d) in enumerate(zip(g1_list, g2_list, d_list)):
                sample_id = i if not sample_ids else sample_ids[i]
                rows.append({
                    RawScoreMetrics.METRIC.value: metric.value if k is None else f"{metric.value}@{k}",
                    RawScoreMetrics.SAMPLE.value: sample_id,
                    RawScoreMetrics.G1_SCORE.value: g1,
                    RawScoreMetrics.G2_SCORE.value: g2,
                    RawScoreMetrics.DELTA.value: d,
                })
            return None

        self._process_metrics(g1_values, add_scores_to_row)
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return df

    def stats_to_csv(self, stats: Dict[ErrorRankingMetrics, Any], filepath: Path) -> pd.DataFrame:
        """
        Saves aggregated statistics to a CSV file.

        Args:
            stats: Dictionary of computed statistics per metric.
            filepath: Path to save CSV.

        Returns:
            DataFrame containing the saved statistics.
        """
        rows = []

        def add_stats_row(metric: ErrorRankingMetrics, k, metric_stats):
            rows.append({
                "metric": metric.value if k is None else f"{metric.value}@{k}",
                **metric_stats
            })
            return None

        self._process_metrics(stats, add_stats_row)
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return df

    def pooled_condition_stats(self, all_deltas: Dict[str, Dict[ErrorRankingMetrics, Any]]) -> Tuple[
        Dict[ErrorRankingMetrics, Any], Dict[ErrorRankingMetrics, Any]]:
        """
        Pools delta values across all conditions and computes summary statistics.

        Args:
            all_deltas: Dictionary mapping condition_name -> deltas dict.

        Returns:
            Tuple of (pooled_stats, pooled_deltas).
        """

        pooled_deltas = self._combine_results_across_conditions(all_deltas)

        def compute_stats(metric, k, vals) -> dict:
            vals = np.array(vals)
            try:
                w = wilcoxon(vals, alternative="greater")
                wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                wilcoxon_stat, wilcoxon_p = np.nan, np.nan

            return {
                PooledStatsMetrics.MEAN.value: float(vals.mean()),
                PooledStatsMetrics.MEDIAN.value: float(np.median(vals)),
                PooledStatsMetrics.WILCOXON_STAT_ONESIDED.value: wilcoxon_stat,
                PooledStatsMetrics.WILCOXON_P_ONESIDED.value: wilcoxon_p,
                PooledStatsMetrics.COUNT_BASE_GT_OTHER.value: int(np.sum(vals > 0)),
                PooledStatsMetrics.COUNT_BASE_LT_OTHER.value: int(np.sum(vals < 0)),
                PooledStatsMetrics.COUNT_EQUAL.value: int(np.sum(vals == 0)),
                PooledStatsMetrics.N_SAMPLES.value: len(vals),
            }

        stats = self._process_metrics(pooled_deltas, compute_stats)
        return stats, pooled_deltas

    def extract_raw_values(self, condition_results: List[Dict[ErrorRankingMetrics, Any]]) -> Tuple[
        Dict[ErrorRankingMetrics, Any], Dict[ErrorRankingMetrics, Any]]:
        """
        Extracts metric values for individual samples from pair comparison results.

        Args:
            condition_results: List of results from compare_rankings() for each sample.

        Returns:
            Tuple of (g1_values, g2_values) dictionaries.
        """
        g1 = self._init_metric_structure()
        g2 = self._init_metric_structure()

        def extract_values(metric, k, result):
            if k is not None:
                append_to_dict_list(g1[metric], k, result.metric_graph1)
                append_to_dict_list(g2[metric], k, result.metric_graph2)
            else:
                g1[metric].append(result.metric_graph1)
                g2[metric].append(result.metric_graph2)
            return None

        for res in condition_results:
            self._process_metrics(res, extract_values)

        return g1, g2

    def extract_metric_deltas(self, condition_results: List[Dict[ErrorRankingMetrics, Any]]) -> Dict[ErrorRankingMetrics, Any]:
        """
        Extracts delta (graph1 - graph2) values from pair comparison results.

        Args:
            condition_results: List of results from compare_rankings() for each sample.

        Returns:
            Dictionary of delta values per metric.
        """
        deltas = self._init_metric_structure()

        def add_delta(metric, k, result):
            d = result.metric_graph1 - result.metric_graph2
            if k is not None:
                append_to_dict_list(deltas[metric], k, d)
            else:
                deltas[metric].append(d)
            return None

        for res in condition_results:
            self._process_metrics(res, add_delta)

        return deltas

    def condition_level_stats(self, deltas: Dict[ErrorRankingMetrics, Any]) -> Dict[ErrorRankingMetrics, Any]:
        """
        Computes statistical tests and summary statistics for each metric's delta values.

        Args:
            deltas: Dictionary of delta values per metric.

        Returns:
            Dictionary of statistics per metric.
        """

        def compute_stats(metric, k, vals) -> dict:
            vals = np.array(vals)
            w = wilcoxon(vals)
            t = ttest_rel(vals, np.zeros_like(vals))

            return {
                ConditionStatsMetrics.MEAN.value: float(vals.mean()),
                ConditionStatsMetrics.MEDIAN.value: float(np.median(vals)),
                ConditionStatsMetrics.WILCOXON_STAT.value: float(w.statistic),
                ConditionStatsMetrics.WILCOXON_P.value: float(w.pvalue),
                ConditionStatsMetrics.TTEST_STAT.value: float(t.statistic),
                ConditionStatsMetrics.TTEST_P.value: float(t.pvalue),
                ConditionStatsMetrics.COUNT_G1_GT_G2.value: int(np.sum(vals > 0)),
                ConditionStatsMetrics.COUNT_G1_LT_G2.value: int(np.sum(vals < 0)),
                ConditionStatsMetrics.COUNT_EQUAL.value: int(np.sum(vals == 0)),
            }

        return self._process_metrics(deltas, compute_stats)

    @staticmethod
    def _process_metrics(data: Dict[ErrorRankingMetrics, Any],
                         fn: Callable[[ErrorRankingMetrics, Any, Any], Any]) -> Dict[ErrorRankingMetrics, Any]:
        """
        Iterates through all metrics in data and applies a function to each.

        Args:
            data: Input data dictionary keyed by metric enum.
            fn: Callable(metric, k_or_none, metric_data) that returns a result.

        Returns:
            Dictionary of results keyed by metric enum.
        """
        results = {}
        for metric in ConfigErrorRankingStep.ALL_METRICS:
            if metric_data := data.get(metric):
                if metric in ConfigErrorRankingStep.K_METRICS:
                    results[metric] = {}
                    for k, k_data in metric_data.items():
                        results[metric][k] = fn(metric, k, k_data)
                else:
                    results[metric] = fn(metric, None, metric_data)
        return results

    @staticmethod
    def _init_metric_structure() -> Dict[ErrorRankingMetrics, Any]:
        """
        Initializes an empty metric structure ({} for K_METRICS, [] for others).

        Returns:
            Dictionary with empty structures for each metric.
        """
        return {metric: ({} if metric in ConfigErrorRankingStep.K_METRICS else [])
                for metric in ConfigErrorRankingStep.ALL_METRICS}

    def _combine_results_across_conditions(self, all_deltas: Dict[str, Dict[ErrorRankingMetrics, Any]]) -> Dict[
        ErrorRankingMetrics, Any]:
        """
        Combines delta values from all conditions into pooled lists.

        Args:
            all_deltas: Dictionary mapping condition_name -> deltas dict.

        Returns:
            Dictionary of pooled delta values per metric.
        """
        pooled_deltas = self._init_metric_structure()

        def extend_pooled(metric, k, vals):
            if k is not None:
                if k not in pooled_deltas[metric]:
                    pooled_deltas[metric][k] = []
                pooled_deltas[metric][k].extend(vals)
            else:
                pooled_deltas[metric].extend(vals)
            return None

        for condition_name, deltas in all_deltas.items():
            self._process_metrics(deltas, extend_pooled)

        return pooled_deltas

    def _save_results(self, output_path: Path, stats: Dict[ErrorRankingMetrics, Any],
                      deltas: Dict[ErrorRankingMetrics, Any], label: str,
                      g1_values: Dict[ErrorRankingMetrics, Any] = None,
                      g2_values: Dict[ErrorRankingMetrics, Any] = None,
                      sample_ids: List[str] = None, is_pooled: bool = False):
        """
        Saves analysis results (stats, plots, CSVs) to the specified path.

        Args:
            output_path: Directory to save results.
            stats: Computed statistics dictionary.
            deltas: Delta values dictionary.
            label: Label for logging output.
            g1_values: Optional graph 1 values for raw scores CSV.
            g2_values: Optional graph 2 values for raw scores CSV.
            sample_ids: Optional sample identifiers for raw scores CSV.
            is_pooled: Whether this is pooled results (adds delta distribution plot).
        """
        output_path.mkdir(parents=True, exist_ok=True)

        if g1_values is not None and g2_values is not None:
            self.raw_scores_to_csv(g1_values, g2_values, deltas, sample_ids=sample_ids,
                                   filepath=output_path / "error-results-individual.csv")

        for metric in ConfigErrorRankingStep.K_METRICS:
            if metric in deltas:
                boxplot_metric_family(deltas[metric], metric.get_printable(),
                                      label = label,
                                      save_path=output_path / f"{metric.name.lower()}_boxplot.png")

        if is_pooled and ErrorRankingMetrics.AP in deltas:
            plot_delta_distribution(deltas[ErrorRankingMetrics.AP],
                                    "Average Precision Δ Distribution (Pooled)",
                                    label = label,
                                    save_path=output_path / "ap_distribution.png")

        self.stats_to_csv(stats, filepath=output_path / "error-results.csv")
        print(f"Saved error results for '{label}' to: {output_path}")
