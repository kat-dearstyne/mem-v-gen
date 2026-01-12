from typing import Dict, Any, List, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import (
    CrossConditionAnalyzeStep
)
from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
    EARLY_LAYER_FRACTION_COL, MAX_LAYER_COL
)
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    PROMPT_TYPE_COL
)
from src.analysis.significance_tester import SignificanceTester
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
        if len(condition_order) < 2:
            return {}

        # Get data for each condition
        condition_data = {}
        for condition in condition_order:
            cond_df = df[df[self.CONDITION_COL] == condition]
            condition_data[condition] = cond_df[EARLY_LAYER_FRACTION_COL].dropna().values

        tester = SignificanceTester(alpha=SIGNIFICANCE_THRESHOLD)

        results = tester.get_descriptive_stats(condition_data)

        sig_df = tester.compare_multiple_groups(condition_data, metric_name=EARLY_LAYER_FRACTION_COL)
        if not sig_df.empty:
            results['_significance_df'] = sig_df

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

        p_value = None
        sig_df = stats_results.get('_significance_df')
        if sig_df is not None and len(condition_order) == 2 and not sig_df.empty:
            p_value = sig_df['mw_p_value'].iloc[0]

        plot_early_layer_mean_comparison(
            df, condition_order, p_value=p_value,
            condition_col=self.CONDITION_COL,
            save_path=save_path / 'early_layer_mean_comparison.png' if save_path else None
        )

        if config_order:
            plot_early_layer_by_config(
                df, condition_order, config_order,
                condition_col=self.CONDITION_COL,
                save_path=save_path / 'early_layer_by_config.png' if save_path else None
            )

        self._generate_prompt_type_visualizations(df, condition_order)

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
