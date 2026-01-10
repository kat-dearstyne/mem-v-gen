from typing import Dict, Any, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import SHARED_FEATURES_KEY
from src.visualizations import plot_shared_feature_metrics


class CrossConditionSharedFeaturesVisualizationStep(CrossConditionAnalyzeStep):
    """
    Cross-condition step for shared feature metrics visualizations.

    Combines shared feature metrics from multiple conditions and generates
    threshold curves and feature presence visualizations.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.SUBGRAPH_FILTER
    RESULTS_SUB_KEY = SHARED_FEATURES_KEY

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[pd.DataFrame]:
        """
        Combines shared feature metrics and generates visualizations.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Combined DataFrame of shared feature metrics, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results, add_condition_as_prompt_type=True)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, config_order = self.get_ordering(combined_df)
        plot_shared_feature_metrics(combined_df, save_dir=self.save_path,
                                    condition_order=condition_order, config_order=config_order)

        return combined_df
