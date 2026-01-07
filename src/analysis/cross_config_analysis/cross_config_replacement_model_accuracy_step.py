from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import zipf_frequency

from src.constants import TOP_K
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ReplacementAccuracyMetrics, \
    ConfigReplacementModelAccuracyStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.utils import load_json, Metrics, create_label_from_conditions, get_as_safe_name
from visualizations import plot_error_hypothesis_combined_boxplot, plot_error_hypothesis_metrics, \
    plot_significance_effect_sizes, plot_omnibus_effect_sizes, plot_per_position_curves, plot_token_complexity

SignificanceStats = namedtuple("SignificanceStats", [
    "group1_mean", "group2_mean", "group1_std", "group2_std", "n_per_group",
    "t_statistic", "t_p_value", "t_significant",
    "mann_whitney_u", "mw_p_value", "mw_significant",
    "cohens_d", "rank_biserial_r"
])

OmnibusSignificanceResult = namedtuple("OmnibusSignificanceResult", [
    "f_statistic", "anova_p_value", "anova_significant", "eta_squared",
    "h_statistic", "kruskal_p_value", "kruskal_significant", "epsilon_squared"
])


class ComplexityMetrics(Metrics):
    """Metrics for measuring token complexity."""
    ZIPF_FREQUENCY = "zipf_frequency"
    TOKEN_LENGTH = "token_length"


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
    EXCLUDE_METRICS = ["top_k_agreement", "replacement_prob_of_original_top"]

    def __init__(self, save_path: Path | None = None):
        """
        Initializes the cross-config replacement model accuracy step.

        Args:
            save_path: Base path for saving results.
        """
        super().__init__(save_path=save_path)

    def run(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]) -> dict | None:
        """
        Runs the analysis step, loading cached results if available.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            Dictionary of analysis results, or None if no cached results exist.
        """
        output_path = self.save_path / "error_hypothesis_analysis.json"
        if output_path.exists() and (results := load_json(output_path)):
            for config, res in results.items():
                if config not in config_results:
                    config_results[config] = {}
                config_results[config][self.CONFIG_RESULTS_KEY] = res

            return self.analyze_results(results=results)
        return None

    def analyze_results(self, results: dict[str, dict[str, Any]],
                        target_condition: str | None = None) -> dict[str, dict]:
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
            return

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
            all_results["omnibus"] = self.df_to_results_dict(omnibus_df, OmnibusSignificanceResult)
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

    def df_to_results_dict(self, df: pd.DataFrame,
                           result_type: type = SignificanceStats) -> dict[str, dict[str, Any]]:
        """
        Convert DataFrame rows to a dictionary keyed by metric name.

        Args:
            df: DataFrame with 'metric' column and result fields.
            result_type: Named tuple type defining the fields to extract.

        Returns:
            Dictionary mapping metric names to their result dictionaries.
        """
        return {row["metric"]: {name: row[name] for name in result_type._fields}
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
                    ) | sig_stats._asdict())

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
            f_stat, anova_p = stats.f_oneway(*groups)

            # Kruskal-Wallis (non-parametric) - tests if distributions differ
            h_stat, kruskal_p = stats.kruskal(*groups)

            # Effect size: eta-squared for ANOVA
            all_values = np.concatenate(groups)
            grand_mean = all_values.mean()
            ss_total = np.sum((all_values - grand_mean) ** 2)
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            # Effect size: epsilon-squared for Kruskal-Wallis
            k, n = len(groups), len(all_values)
            epsilon_squared = (h_stat - k + 1) / (n - k)

            results_rows.append(
                dict(metric=metric.value,
                     conditions=", ".join(conditions),
                     n_groups=len(conditions)) |
                OmnibusSignificanceResult(
                    f_statistic=f_stat,
                    anova_p_value=anova_p,
                    anova_significant=self.is_significant(anova_p),
                    eta_squared=eta_squared,
                    h_statistic=h_stat,
                    kruskal_p_value=kruskal_p,
                    kruskal_significant=self.is_significant(kruskal_p),
                    epsilon_squared=epsilon_squared,
                )._asdict())

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
            df[metric.value] = df["original_top_token"].apply(
                lambda t: self.get_token_complexity(t)[metric]
            )
            mem_values = self._get_metric_values_for_condition(df, metric, target_condition)

            for other_cond in other_conditions:
                other_values = self._get_metric_values_for_condition(df, metric, other_cond)

                sig_stats = self.compute_significance_stats(mem_values, other_values, alternative='two-sided')

                results_rows.append(dict(
                    metric=metric.value,
                    comparison=create_label_from_conditions(target_condition, other_cond),
                ) | sig_stats._asdict())

        plot_token_complexity(df, self.save_path, [target_condition] + other_conditions)

        # Save complexity data
        complexity_dir = self.save_path / "token_complexity"
        complexity_dir.mkdir(parents=True, exist_ok=True)

        complexity_df = df[["condition", "config", "original_top_token"] + [e.value for e in ComplexityMetrics]]
        complexity_df.to_csv(complexity_dir / "token_complexity.csv", index=False)

        results_df = self.apply_bh_correction(pd.DataFrame(results_rows))
        results_df.to_csv(complexity_dir / "complexity_significance.csv", index=False)

        return results_df

    def compute_significance_stats(self, group1: np.ndarray, group2: np.ndarray,
                                   alternative: str = 'two-sided') -> SignificanceStats:
        """
        Compute common significance test statistics between two groups.

        Args:
            group1: First group of values.
            group2: Second group of values.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            SignificanceStats with all computed statistics.
        """
        n1, n2 = len(group1), len(group2)

        # T-test (parametric)
        t_stat, t_p = stats.ttest_ind(group1, group2, alternative=alternative, equal_var=False)

        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p = stats.mannwhitneyu(group1, group2, alternative=alternative)

        # Effect sizes
        cohens_d = self.compute_cohens_d(group1, group2)
        rank_biserial = self.compute_rank_biserial(mw_stat, n1, n2)

        return SignificanceStats(
            group1_mean=group1.mean(),
            group2_mean=group2.mean(),
            group1_std=np.std(group1, ddof=1),
            group2_std=np.std(group2, ddof=1),
            n_per_group=n1,
            t_statistic=t_stat,
            t_p_value=t_p,
            t_significant=t_p < 0.05,
            mann_whitney_u=mw_stat,
            mw_p_value=mw_p,
            mw_significant=mw_p < 0.05,
            cohens_d=cohens_d,
            rank_biserial_r=rank_biserial,
        )

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
        for metric_name in results_df.columns:
            if metric_name.endswith("p_value"):
                p_values = results_df[metric_name].values
                adjusted = stats.false_discovery_control(p_values, method='bh')
                column_name = f"{metric_name}_bh"
                results_df[column_name] = adjusted
                sign_col_name = column_name.replace("p_value", "significant")
                results_df[sign_col_name] = CrossConfigReplacementModelAccuracyStep.is_significant(adjusted)
        return results_df

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
        return p_value < significance_threshold

    @staticmethod
    def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size using pooled standard deviation.

        Uses the average of variances (Welch-consistent, equal-n friendly).

        Args:
            group1: First group of values.
            group2: Second group of values.

        Returns:
            Cohen's d effect size.
        """
        sx = np.std(group1, ddof=1)
        sy = np.std(group2, ddof=1)
        denom = np.sqrt((sx ** 2 + sy ** 2) / 2)
        return (group1.mean() - group2.mean()) / denom if denom > 0 else 0.0

    @staticmethod
    def compute_rank_biserial(mw_stat: float, n1: int, n2: int) -> float:
        """
        Compute rank-biserial correlation from Mann-Whitney U statistic.

        Args:
            mw_stat: Mann-Whitney U statistic.
            n1: Size of first group.
            n2: Size of second group.

        Returns:
            Rank-biserial correlation coefficient (r = 2U/(n1*n2) - 1).
        """
        return (2 * mw_stat) / (n1 * n2) - 1

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

    def _to_df(self, results: dict[str, dict[str, Any]]) -> pd.DataFrame:
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
                rows.append({
                    "condition": condition,
                    "config": config_name,
                    **metrics
                })
        return pd.DataFrame(rows)
