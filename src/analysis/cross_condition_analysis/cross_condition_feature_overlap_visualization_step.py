from typing import Dict, Any, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.visualizations import plot_combined_metrics


class CrossConditionFeatureOverlapVisualizationStep(CrossConditionAnalyzeStep):
    """
    Cross-condition step for feature overlap metrics visualizations.

    Combines feature sharing metrics (unique/shared fractions) from multiple
    conditions and generates comparison visualizations.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.FEATURE_OVERLAP
    RESULTS_SUB_KEY = None  # Direct DataFrame, no sub-key

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[pd.DataFrame]:
        """
        Combines feature overlap metrics and generates visualizations.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Combined DataFrame of feature overlap metrics, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, config_order = self.get_ordering(combined_df)
        plot_combined_metrics(combined_df, save_dir=self.save_path,
                              condition_order=condition_order, config_order=config_order)

        return combined_df
