from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import CONFIG_NAME_COL, PROMPT_TYPE_COL
from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import EARLY_LAYER_FRACTION_COL
from src.visualizations import (
    plot_early_layer_boxplot,
    plot_early_layer_mean_comparison,
    plot_early_layer_by_config
)

# Output filenames
EARLY_LAYER_COMPARISON_FILENAME = "early_layer_comparison.csv"
EARLY_LAYER_STATS_FILENAME = "early_layer_stats.csv"


class CrossConditionEarlyLayerStep(CrossConditionAnalyzeStep):
    """
    Cross-condition step for comparing early layer contribution between conditions.

    Performs statistical tests to determine if conditions differ significantly
    in their early layer contribution fractions, and visualizes the results.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION
    RESULTS_SUB_KEY = None

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[Dict[str, Any]]:
        """
        Compares early layer contribution between conditions.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Dictionary with combined DataFrame and statistical results, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results, add_condition_as_prompt_type=False)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, config_order = self.get_ordering(combined_df)

        # Calculate statistics
        stats_results = self._calculate_statistics(combined_df, condition_order)

        # Generate visualizations
        self._generate_visualizations(combined_df, stats_results, condition_order, config_order)

        # Save results
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(self.save_path / EARLY_LAYER_COMPARISON_FILENAME, index=False)

            stats_df = pd.DataFrame([stats_results])
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
            Dictionary with statistical test results.
        """
        results = {}

        if len(condition_order) < 2:
            return results

        # Get data for each condition
        condition_data = {}
        for condition in condition_order:
            cond_df = df[df[PROMPT_TYPE_COL] == condition]
            condition_data[condition] = cond_df[EARLY_LAYER_FRACTION_COL].dropna().values

        # Calculate descriptive stats per condition
        for condition, values in condition_data.items():
            results[f'{condition}_mean'] = float(np.mean(values)) if len(values) > 0 else np.nan
            results[f'{condition}_std'] = float(np.std(values)) if len(values) > 0 else np.nan
            results[f'{condition}_n'] = len(values)

        # Pairwise comparisons between conditions
        conditions = list(condition_data.keys())
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                vals1 = condition_data[cond1]
                vals2 = condition_data[cond2]

                if len(vals1) < 2 or len(vals2) < 2:
                    continue

                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
                    results[f'{cond1}_vs_{cond2}_mannwhitney_u'] = float(u_stat)
                    results[f'{cond1}_vs_{cond2}_mannwhitney_p'] = float(u_pval)
                except Exception:
                    pass

                # Independent t-test
                try:
                    t_stat, t_pval = stats.ttest_ind(vals1, vals2)
                    results[f'{cond1}_vs_{cond2}_ttest_t'] = float(t_stat)
                    results[f'{cond1}_vs_{cond2}_ttest_p'] = float(t_pval)
                except Exception:
                    pass

                # Effect size (Cohen's d)
                try:
                    pooled_std = np.sqrt(((len(vals1)-1)*np.var(vals1) + (len(vals2)-1)*np.var(vals2)) /
                                         (len(vals1) + len(vals2) - 2))
                    if pooled_std > 0:
                        cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std
                        results[f'{cond1}_vs_{cond2}_cohens_d'] = float(cohens_d)
                except Exception:
                    pass

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
            save_path=save_path / 'early_layer_mean_comparison.png' if save_path else None
        )

        # Line plot by config
        if config_order:
            plot_early_layer_by_config(
                df, condition_order, config_order,
                save_path=save_path / 'early_layer_by_config.png' if save_path else None
            )
