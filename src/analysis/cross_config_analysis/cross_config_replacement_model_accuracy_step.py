from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import zipf_frequency

from src.constants import TOP_K
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ConfigReplacementModelAccuracyStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.analysis.significance_tester import SignificanceTester
from src.metrics import (
    Metrics, ReplacementAccuracyMetrics, ComplexityMetrics,
    SignificanceMetrics, OmnibusSignificanceMetrics
)
from src.utils import load_json, save_json, create_label_from_conditions, get_as_safe_name
from src.visualizations import plot_error_hypothesis_combined_boxplot, plot_error_hypothesis_metrics, \
    plot_significance_effect_sizes, plot_omnibus_effect_sizes, plot_per_position_curves, plot_token_complexity


class CrossConfigReplacementModelAccuracyStep(CrossConfigAnalyzeStep):
    """
    Cross-config analysis step for replacement model accuracy.

    Compares replacement model performance across conditions, computing pairwise
    and omnibus significance tests with effect sizes. Also analyzes token complexity
    as a potential confound.
    """
    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.REPLACEMENT_MODEL
    TARGET_CONDITION = "memorized"
    SIGNIFICANCE_THRESHOLD = 0.05
    EXCLUDE_METRICS = [
        ReplacementAccuracyMetrics.TOP_K_AGREEMENT.value,
        ReplacementAccuracyMetrics.REPLACEMENT_PROB_OF_ORIGINAL_TOP.value
    ]

    def __init__(self, save_path: Path | None = None):
        """
        Initializes the cross-config replacement model accuracy step.

        Args:
            save_path: Base path for saving results.
        """
        super().__init__(save_path=save_path)

    def run(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]) -> dict | None:
        """
        Runs the analysis step, using passed-in results or loading cached results.

        If config_results contains replacement model data, extracts it, saves to JSON,
        and runs analysis. Otherwise, attempts to load from cached JSON file.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            Dictionary of analysis results, or None if no results available.
        """
        if self.save_path is None:
            return None

        # Check if results were passed in
        results = self._extract_results(config_results)

        if results:
            self._save_results(config_results)
            return self.analyze_results(results=results)

        # Try to load from file
        output_path = self.save_path / "error_hypothesis_analysis.json"
        if output_path.exists() and (results := load_json(output_path)):
            for config, res in results.items():
                if config not in config_results:
                    config_results[config] = {}
                config_results[config][self.CONFIG_RESULTS_KEY] = res

            return self.analyze_results(results=results)
        return None

    def _extract_results(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]
                         ) -> dict[str, dict[str, Any]] | None:
        """
        Extract replacement model results from config_results if present.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            Extracted results in format {config_name: {condition: metrics}}, or None.
        """
        results = {}
        for config_name, step_results in config_results.items():
            if self.CONFIG_RESULTS_KEY in step_results:
                step_data = step_results[self.CONFIG_RESULTS_KEY]
                if step_data:
                    results[config_name] = step_data
        return results if results else None

    def _save_results(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]) -> None:
        """
        Save results to JSON file.

        Converts enum keys to string values for JSON serialization.

        Args:
            config_results: Dictionary mapping config names to their per-step results.
        """
        self.save_path.mkdir(parents=True, exist_ok=True)
        output_path = self.save_path / "error_hypothesis_analysis.json"

        serializable_results = {}
        for config_name, step_results in config_results.items():
            if self.CONFIG_RESULTS_KEY not in step_results:
                continue
            condition_metrics = step_results[self.CONFIG_RESULTS_KEY]
            serializable_results[config_name] = {
                condition: {
                    (k.value if hasattr(k, 'value') else k): v
                    for k, v in metrics.items()
                }
                for condition, metrics in condition_metrics.items()
            }

        save_json(serializable_results, output_path)
        print(f"Replacement model results saved to: {output_path}")

    def analyze_results(self, results: dict[str, dict[str, Any]],
                        target_condition: str | None = None) -> Optional[dict[str, dict]]:
        """
        Analyze results and create visualizations comparing conditions.

        Args:
            results: Dictionary mapping config names to condition metrics.
            target_condition: Condition to compare against others (default: TARGET_CONDITION).

        Returns:
            Dictionary containing pairwise and omnibus significance results.
        """
        # Convert to DataFrame for easier plotting
        df = self._to_df(results)

        if df.empty:
            print("No results to analyze")
            return None

        all_results = {}
        conditions = df["condition"].unique().tolist()
        target_condition = self.TARGET_CONDITION if not target_condition else target_condition
        assert target_condition in conditions, f"Unknown condition {target_condition}"
        other_conditions = [c for c in conditions if c != target_condition]

        plot_error_hypothesis_metrics(df, self.save_path, top_k=TOP_K)
        plot_error_hypothesis_combined_boxplot(df, conditions=conditions,
                                               save_path=self.save_path / "combined_metrics_boxplot.png")

        pairwise_df = self.compute_pairwise_significance(df, target_condition=target_condition,
                                                         other_conditions=other_conditions, test_target_worse=True)
        omnibus_df = self.compute_omnibus_significance(df, conditions=conditions)

        # Significance effect size visualizations
        sig_viz_dir = self.save_path / "significance_viz"
        sig_viz_dir.mkdir(parents=True, exist_ok=True)

        if not pairwise_df.empty:
            pairwise_df.to_csv(self.save_path / "significance_analysis.csv", index=False)
            for comparison in pairwise_df["comparison"].unique():
                comparison_df = self._add_comparison_to_results(all_results, comparison, pairwise_df)
                plot_significance_effect_sizes(
                    comparison_df,
                    title=comparison.replace("_", " ").title(),
                    save_path=sig_viz_dir / f"{get_as_safe_name(comparison)}_effect_sizes.png",
                    exclude_metrics=self.EXCLUDE_METRICS,
                )

        if not omnibus_df.empty:
            omnibus_df.to_csv(self.save_path / "omnibus_significance_analysis.csv", index=False)
            all_results["omnibus"] = self.df_to_results_dict(omnibus_df, OmnibusSignificanceMetrics)
            plot_omnibus_effect_sizes(
                omnibus_df,
                title="Omnibus Tests (ANOVA / Kruskal-Wallis)",
                save_path=sig_viz_dir / "omnibus_effect_sizes.png",
                exclude_metrics=self.EXCLUDE_METRICS,
            )

        # Per-position curve visualizations
        plot_per_position_curves(results, self.save_path, conditions)

        # Token complexity analysis
        token_complexity_df = self.analyze_token_complexity(df, target_condition=target_condition,
                                                            other_conditions=other_conditions)
        if not token_complexity_df.empty:
            for comparison in token_complexity_df["comparison"].unique():
                self._add_comparison_to_results(all_results, comparison, token_complexity_df)

        print(f"\nResults saved to: {self.save_path}")
        return all_results

    def _add_comparison_to_results(self, all_results: dict[str, dict],
                                   comparison: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comparison results to the all_results dictionary.

        Args:
            all_results: Dictionary to update with comparison results.
            comparison: Comparison label (e.g., 'memorized vs. random').
            df: DataFrame containing results for all comparisons.

        Returns:
            Filtered DataFrame containing only rows for the specified comparison.
        """
        comparison_df = df[df["comparison"] == comparison]
        if comparison not in all_results:
            all_results[comparison] = {}
        all_results[comparison].update(self.df_to_results_dict(comparison_df))

        return comparison_df

    @staticmethod
    def df_to_results_dict(df: pd.DataFrame,
                           metrics_enum: type[Metrics] = SignificanceMetrics) -> dict[str, dict[str, Any]]:
        """
        Convert DataFrame rows to a dictionary keyed by metric name.

        Args:
            df: DataFrame with 'metric' column and result fields.
            metrics_enum: Metrics enum class defining the fields to extract.

        Returns:
            Dictionary mapping metric names to their result dictionaries.
        """
        field_names = [m.value for m in metrics_enum]
        return {row["metric"]: {name: row[name] for name in field_names}
                for _, row in df.iterrows()}

    def compute_pairwise_significance(self, df: pd.DataFrame, target_condition: str,
                                      other_conditions: list[str],
                                      test_target_worse: bool = True) -> pd.DataFrame:
        """
        Compute pairwise significance tests between target and other conditions.

        Args:
            df: DataFrame with 'condition' column and metric columns.
            target_condition: The condition to compare against others.
            other_conditions: List of conditions to compare target against.
            test_target_worse: If True, test if target performs worse (lower similarity,
                higher KL). If False, test if target performs better.

        Returns:
            DataFrame with significance results including BH-corrected p-values.
        """

        result_rows = []

        for metric in self._get_metrics_for_significance(df):

            is_higher_better = metric in ConfigReplacementModelAccuracyStep.HIGHER_IS_BETTER_METRICS
            target_values = self._get_metric_values_for_condition(df, metric, target_condition)

            for other_cond in other_conditions:
                other_values = self._get_metric_values_for_condition(df, metric, other_cond)

                # Determine test direction
                if test_target_worse:
                    alternative = 'less' if is_higher_better else 'greater'
                else:
                    alternative = 'greater' if is_higher_better else 'less'

                sig_stats = self.compute_significance_stats(target_values, other_values, alternative)
                result_rows.append(
                    dict(
                        metric=metric.value,
                        comparison=create_label_from_conditions(target_condition, other_cond),
                        target_condition=target_condition,
                        other_condition=other_cond,
                        test_direction="target_worse" if test_target_worse else "target_better",
                    ) | sig_stats)

        results_df = self.apply_bh_correction(pd.DataFrame(result_rows))
        return results_df

    def compute_omnibus_significance(self, df: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
        """
        Compute omnibus significance tests (ANOVA and Kruskal-Wallis) across all conditions.

        These tests determine whether there is a significant difference among ANY of the
        groups, without specifying which groups differ. Use alongside pairwise comparisons.

        Args:
            df: DataFrame with 'condition' column and metric columns.
            conditions: List of conditions to compare.

        Returns:
            DataFrame with omnibus test results (one row per metric).
        """
        if len(conditions) < 2:
            return pd.DataFrame()

        results_rows = []

        for metric in self._get_metrics_for_significance(df):
            # Get values for each condition
            groups = [self._get_metric_values_for_condition(df, metric, cond) for cond in conditions]

            # ANOVA (parametric) - tests if means differ
            try:
                f_stat, anova_p = stats.f_oneway(*groups)
            except ValueError:
                f_stat, anova_p = np.nan, np.nan

            # Kruskal-Wallis (non-parametric) - tests if distributions differ
            try:
                h_stat, kruskal_p = stats.kruskal(*groups)
            except ValueError:
                h_stat, kruskal_p = np.nan, np.nan

            # Effect size: eta-squared for ANOVA
            all_values = np.concatenate(groups)
            grand_mean = all_values.mean()
            ss_total = np.sum((all_values - grand_mean) ** 2)
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            # Effect size: epsilon-squared for Kruskal-Wallis
            k, n = len(groups), len(all_values)
            epsilon_squared = (h_stat - k + 1) / (n - k) if n > k else np.nan

            results_rows.append({
                "metric": metric.value,
                "conditions": ", ".join(conditions),
                "n_groups": len(conditions),
                OmnibusSignificanceMetrics.F_STATISTIC.value: f_stat,
                OmnibusSignificanceMetrics.ANOVA_P_VALUE.value: anova_p,
                OmnibusSignificanceMetrics.ANOVA_SIGNIFICANT.value: self.is_significant(anova_p),
                OmnibusSignificanceMetrics.ETA_SQUARED.value: eta_squared,
                OmnibusSignificanceMetrics.H_STATISTIC.value: h_stat,
                OmnibusSignificanceMetrics.KRUSKAL_P_VALUE.value: kruskal_p,
                OmnibusSignificanceMetrics.KRUSKAL_SIGNIFICANT.value: self.is_significant(kruskal_p),
                OmnibusSignificanceMetrics.EPSILON_SQUARED.value: epsilon_squared,
            })

        return pd.DataFrame(results_rows)

    def analyze_token_complexity(self, df: pd.DataFrame, target_condition: str,
                                 other_conditions: list[str]) -> pd.DataFrame:
        """
        Analyze token complexity across conditions to check for confounds.

        Computes complexity metrics (Zipf frequency, token length), creates
        visualizations, runs statistical tests, and saves results.

        Args:
            df: DataFrame with columns: condition, config, original_top_token.
            target_condition: The condition to compare against others.
            other_conditions: List of conditions to compare target against.

        Returns:
            DataFrame with complexity significance results.
        """
        # Statistical tests - pairwise comparisons like metric significance
        results_rows = []

        for metric in ComplexityMetrics:
            df[metric.value] = df[ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value].apply(
                lambda t: self.get_token_complexity(t)[metric]
            )
            mem_values = self._get_metric_values_for_condition(df, metric, target_condition)

            for other_cond in other_conditions:
                other_values = self._get_metric_values_for_condition(df, metric, other_cond)

                sig_stats = self.compute_significance_stats(mem_values, other_values, alternative='two-sided')

                results_rows.append(dict(
                    metric=metric.value,
                    comparison=create_label_from_conditions(target_condition, other_cond),
                ) | sig_stats)

        plot_token_complexity(df, self.save_path, [target_condition] + other_conditions)

        # Save complexity data
        complexity_dir = self.save_path / "token_complexity"
        complexity_dir.mkdir(parents=True, exist_ok=True)

        original_top_token_col = ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value
        complexity_df = df[["condition", "config", original_top_token_col] + [e.value for e in ComplexityMetrics]]
        complexity_df.to_csv(complexity_dir / "token_complexity.csv", index=False)

        results_df = self.apply_bh_correction(pd.DataFrame(results_rows))
        results_df.to_csv(complexity_dir / "complexity_significance.csv", index=False)

        return results_df

    def compute_significance_stats(self, group1: np.ndarray, group2: np.ndarray,
                                   alternative: str = 'two-sided') -> dict[str, Any]:
        """
        Compute common significance test statistics between two groups.

        Args:
            group1: First group of values.
            group2: Second group of values.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary with SignificanceMetrics values as keys.
        """
        tester = SignificanceTester(alpha=self.SIGNIFICANCE_THRESHOLD)
        return tester.compute_stats(group1, group2, alternative)

    @staticmethod
    def get_token_complexity(token: str) -> dict[ComplexityMetrics, float]:
        """
        Get complexity metrics for a token.

        Args:
            token: The token string (may include leading space from tokenizer)

        Returns:
            Dictionary with zipf_frequency and token_length
        """
        clean_token = token.strip()

        freq = zipf_frequency(clean_token, 'en') if clean_token else 0.0

        return {
            ComplexityMetrics.ZIPF_FREQUENCY: freq,
            ComplexityMetrics.TOKEN_LENGTH: len(clean_token),
        }

    @staticmethod
    def apply_bh_correction(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Benjamini-Hochberg FDR correction to p-values.

        Adds BH-corrected p-value and significance columns for each p-value column.

        Args:
            results_df: DataFrame with columns ending in 'p_value'.

        Returns:
            DataFrame with added *_bh and *_significant_bh columns.
        """
        return SignificanceTester.apply_bh_correction(results_df)

    @staticmethod
    def is_significant(p_value: float | np.ndarray,
                       significance_threshold: float = SIGNIFICANCE_THRESHOLD) -> bool | np.ndarray:
        """
        Check if p-value(s) are below significance threshold.

        Args:
            p_value: Single p-value or array of p-values.
            significance_threshold: Threshold for significance (default: 0.05).

        Returns:
            Boolean or array of booleans indicating significance.
        """
        return SignificanceTester.is_significant(p_value, significance_threshold)

    @staticmethod
    def _get_metric_values_for_condition(df: pd.DataFrame, metric: Metrics | str,
                                         cond: str) -> np.ndarray:
        """
        Extract metric values for a specific condition from DataFrame.

        Args:
            df: DataFrame with 'condition' column and metric columns.
            metric: Metric enum or string column name.
            cond: Condition to filter by.

        Returns:
            Array of metric values for the specified condition.
        """
        metric = metric.value if isinstance(metric, Metrics) else metric
        return df[df["condition"] == cond][metric].values

    @staticmethod
    def _get_metrics_for_significance(df: pd.DataFrame) -> list[ReplacementAccuracyMetrics]:
        """
        Get metrics that are available in the DataFrame and have defined directionality.

        Args:
            df: DataFrame with metric columns.

        Returns:
            List of metrics available for significance testing.
        """
        return [metric for metric in ReplacementAccuracyMetrics if metric.value in df.columns and
                (metric in ConfigReplacementModelAccuracyStep.HIGHER_IS_BETTER_METRICS or
                 metric in ConfigReplacementModelAccuracyStep.LOWER_IS_BETTER_METRICS)]

    @staticmethod
    def _to_df(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """
        Convert nested results dictionary to a flat DataFrame.

        Args:
            results: Dictionary mapping config names to condition metrics.

        Returns:
            DataFrame with columns: condition, config, and metric columns.
        """
        rows = []
        for config_name, condition_metrics in results.items():
            for condition, metrics in condition_metrics.items():
                # Convert enum keys to string values
                metrics_with_str_keys = {
                    (k.value if hasattr(k, 'value') else k): v
                    for k, v in metrics.items()
                }
                rows.append({
                    "condition": condition,
                    "config": config_name,
                    **metrics_with_str_keys
                })
        return pd.DataFrame(rows)
