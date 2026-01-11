from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.visualizations import plot_l0_per_layer_by_condition

L0_COMPARISON_FILENAME = "l0_comparison.csv"


class CrossConditionL0Step(CrossConditionAnalyzeStep):
    """
    Cross-condition step for visualizing L0 (active features) across conditions.

    Combines L0 data from multiple conditions and creates visualizations
    showing L0 per layer with conditions side by side.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL
    RESULTS_SUB_KEY = "df"

    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Optional[pd.DataFrame]:
        """
        Combines L0 data from all conditions and creates visualizations.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.

        Returns:
            Combined DataFrame with condition column, or None if no data.
        """
        combined_df = self.combine_condition_dataframes(condition_results)

        if combined_df is None or combined_df.empty:
            return None

        condition_order, _ = self.get_ordering(combined_df)

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(self.save_path / L0_COMPARISON_FILENAME, index=False)

            plot_l0_per_layer_by_condition(
                combined_df,
                condition_order=condition_order,
                condition_col=self.CONDITION_COL,
                save_path=self.save_path / "l0_per_layer_by_condition.png"
            )

            print(f"Saved L0 cross-condition results to: {self.save_path}")

        return combined_df
