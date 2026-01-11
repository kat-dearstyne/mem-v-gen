from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import (
    CrossConditionAnalyzeStep
)
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CONFIG_NAME_COL, PROMPT_TYPE_COL
)
from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
    EARLY_LAYER_FRACTION_COL, MAX_LAYER_COL
)
from src.metrics import EarlyLayerMetrics
from src.visualizations import (
    plot_early_layer_boxplot,
    plot_early_layer_mean_comparison,
    plot_early_layer_by_config,
    plot_significance_effect_sizes,
    plot_early_layer_threshold_comparison,
    plot_early_layer_by_prompt_type,
    plot_early_layer_prompt_type_lines
)

# Output filenames
EARLY_LAYER_COMPARISON_FILENAME = "early_layer_comparison.csv"
EARLY_LAYER_STATS_FILENAME = "early_layer_stats.csv"
EARLY_LAYER_SIGNIFICANCE_FILENAME = "early_layer_significance.csv"

SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_PRIMARY_THRESHOLD = 2


class CrossConditionEarlyLayerStep(CrossConditionAnalyzeStep):
    """
    Cross-condition step for comparing early layer contribution between conditions.

    Performs statistical tests to determine if conditions differ significantly
    in their early layer contribution fractions, and visualizes the results.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION
    RESULTS_SUB_KEY = None

    def __init__(self, primary_threshold: int = DEFAULT_PRIMARY_THRESHOLD, **kwargs):
        """
        Args:
            primary_threshold: Max layer threshold to use for primary analysis when
                multiple thresholds are present. Defaults to 2.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self.primary_threshold = primary_threshold

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[Dict[str, Any]]:
        """
        Compares early layer contribution between conditions.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Dictionary with combined DataFrame and statistical results, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, config_order = self.get_ordering(combined_df)

        # Check if we have multiple max_layer values
        max_layers = sorted(combined_df[MAX_LAYER_COL].unique()) if MAX_LAYER_COL in combined_df.columns else [self.primary_threshold]
        has_multiple_thresholds = len(max_layers) > 1

        # Filter to primary threshold for statistics and main visualizations
        if has_multiple_thresholds:
            primary_df = combined_df[combined_df[MAX_LAYER_COL] == self.primary_threshold].copy()
        else:
            primary_df = combined_df

        # Calculate statistics on primary threshold data
        stats_results = self._calculate_statistics(primary_df, condition_order)

        # Generate visualizations for primary threshold
        self._generate_visualizations(primary_df, stats_results, condition_order, config_order)

        # Generate threshold comparison visualization if multiple thresholds
        if has_multiple_thresholds:
            plot_early_layer_threshold_comparison(
                combined_df, condition_order, max_layers,
                condition_col=self.CONDITION_COL,
                save_path=self.save_path / 'early_layer_threshold_comparison.png' if self.save_path else None
            )

        # Save results
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(self.save_path / EARLY_LAYER_COMPARISON_FILENAME, index=False)

            # Filter out DataFrame from stats before saving
            stats_to_save = {k: v for k, v in stats_results.items() if not isinstance(v, pd.DataFrame)}
            stats_df = pd.DataFrame([stats_to_save])
            stats_df.to_csv(self.save_path / EARLY_LAYER_STATS_FILENAME, index=False)

        return {
            'data': combined_df,
            'statistics': stats_results
        }

    def _calculate_statistics(self, df: pd.DataFrame, condition_order: List[str]) -> Dict[str, Any]:
        """
        Calculates statistical tests comparing conditions.

        Args:
            df: Combined DataFrame with early layer fractions.
            condition_order: Order of conditions.

        Returns:
            Dictionary with statistical test results and significance DataFrame.
        """
        results = {}

        if len(condition_order) < 2:
            return results

        # Get data for each condition
        condition_data = {}
        for condition in condition_order:
            cond_df = df[df[self.CONDITION_COL] == condition]
            condition_data[condition] = cond_df[EARLY_LAYER_FRACTION_COL].dropna().values

        # Calculate descriptive stats per condition
        for condition, values in condition_data.items():
            results[f'{condition}_mean'] = float(np.mean(values)) if len(values) > 0 else np.nan
            results[f'{condition}_std'] = float(np.std(values)) if len(values) > 0 else np.nan
            results[f'{condition}_n'] = len(values)

        # Pairwise comparisons between conditions
        conditions = list(condition_data.keys())
        significance_rows = []

        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                vals1 = condition_data[cond1]
                vals2 = condition_data[cond2]

                if len(vals1) < 2 or len(vals2) < 2:
                    continue

                row = {'metric': EARLY_LAYER_FRACTION_COL, 'comparison': f'{cond1} vs {cond2}'}

                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    row['mw_p_value'] = float(u_pval)
                    row['mw_significant'] = u_pval < SIGNIFICANCE_THRESHOLD
                    # Rank-biserial correlation
                    n1, n2 = len(vals1), len(vals2)
                    row['rank_biserial_r'] = (2 * u_stat) / (n1 * n2) - 1
                    results[f'{cond1}_vs_{cond2}_mannwhitney_p'] = float(u_pval)
                except Exception:
                    pass

                # Independent t-test
                try:
                    t_stat, t_pval = stats.ttest_ind(vals1, vals2, equal_var=False)
                    row['t_p_value'] = float(t_pval)
                    row['t_significant'] = t_pval < SIGNIFICANCE_THRESHOLD
                    results[f'{cond1}_vs_{cond2}_ttest_p'] = float(t_pval)
                except Exception:
                    pass

                # Effect size (Cohen's d)
                try:
                    sx, sy = np.std(vals1, ddof=1), np.std(vals2, ddof=1)
                    denom = np.sqrt((sx ** 2 + sy ** 2) / 2)
                    row['cohens_d'] = (np.mean(vals1) - np.mean(vals2)) / denom if denom > 0 else 0.0
                    results[f'{cond1}_vs_{cond2}_cohens_d'] = row['cohens_d']
                except Exception:
                    pass

                significance_rows.append(row)

        if significance_rows:
            results['_significance_df'] = pd.DataFrame(significance_rows)

        return results

    def _generate_visualizations(self, df: pd.DataFrame, stats_results: Dict[str, Any],
                                  condition_order: List[str], config_order: List[str]) -> None:
        """
        Generates visualizations for early layer contribution comparison.

        Args:
            df: Combined DataFrame with early layer fractions.
            stats_results: Statistical test results.
            condition_order: Order of conditions.
            config_order: Order of configs.
        """
        save_path = self.save_path

        # Boxplot by condition
        plot_early_layer_boxplot(
            df, condition_order,
            condition_col=self.CONDITION_COL,
            save_path=save_path / 'early_layer_boxplot.png' if save_path else None
        )

        # Get p-value for significance annotation
        p_value = None
        if len(condition_order) == 2:
            cond1, cond2 = condition_order
            p_key = f'{cond1}_vs_{cond2}_mannwhitney_p'
            if p_key not in stats_results:
                p_key = f'{cond2}_vs_{cond1}_mannwhitney_p'
            p_value = stats_results.get(p_key)

        # Bar chart with means and error bars
        plot_early_layer_mean_comparison(
            df, condition_order, p_value=p_value,
            condition_col=self.CONDITION_COL,
            save_path=save_path / 'early_layer_mean_comparison.png' if save_path else None
        )

        # Line plot by config
        if config_order:
            plot_early_layer_by_config(
                df, condition_order, config_order,
                condition_col=self.CONDITION_COL,
                save_path=save_path / 'early_layer_by_config.png' if save_path else None
            )

        # Prompt type comparisons across conditions
        self._generate_prompt_type_visualizations(df, condition_order)

        # Significance effect size visualization
        sig_df = stats_results.get('_significance_df')
        if sig_df is not None and save_path:
            sig_df.to_csv(save_path / EARLY_LAYER_SIGNIFICANCE_FILENAME, index=False)
            plot_significance_effect_sizes(
                sig_df,
                title='Early Layer Contribution',
                save_path=save_path / 'early_layer_effect_sizes.png',
                t_sig_col='t_significant',
                mw_sig_col='mw_significant'
            )

    def _generate_prompt_type_visualizations(self, df: pd.DataFrame,
                                              condition_order: List[str]) -> None:
        """
        Generates visualizations comparing conditions across prompt types.

        Args:
            df: Combined DataFrame with early layer fractions.
            condition_order: Order of conditions.
        """
        if PROMPT_TYPE_COL not in df.columns:
            return

        prompt_type_order = sorted(df[PROMPT_TYPE_COL].unique())
        if not prompt_type_order:
            return

        save_path = self.save_path

        plot_early_layer_by_prompt_type(
            df, condition_order, prompt_type_order,
            condition_col=self.CONDITION_COL,
            prompt_type_col=PROMPT_TYPE_COL,
            save_path=save_path / 'early_layer_by_prompt_type.png' if save_path else None
        )
        plot_early_layer_prompt_type_lines(
            df, condition_order, prompt_type_order,
            condition_col=self.CONDITION_COL,
            prompt_type_col=PROMPT_TYPE_COL,
            save_path=save_path / 'early_layer_prompt_type_lines.png' if save_path else None
        )
