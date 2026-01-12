from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
    L0_VALUE_COL, L0_NORMALIZED_COL
)
from src.visualizations import plot_l0_per_layer_by_condition, plot_l0_per_layer_line

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

            self._generate_l0_plots(combined_df, condition_order, L0_VALUE_COL, "l0")

            if L0_NORMALIZED_COL in combined_df.columns and combined_df[L0_NORMALIZED_COL].notna().any():
                self._generate_l0_plots(combined_df, condition_order, L0_NORMALIZED_COL, "l0_normalized")

            print(f"Saved L0 cross-condition results to: {self.save_path}")

        return combined_df

    def _generate_l0_plots(self, df: pd.DataFrame, condition_order: List[str],
                           l0_col: str, prefix: str) -> None:
        """
        Generates bar and line plots for L0 data.

        Args:
            df: DataFrame with L0 data.
            condition_order: Order of conditions for plotting.
            l0_col: Column name containing L0 values.
            prefix: Filename prefix for saved plots.
        """
        plot_l0_per_layer_by_condition(
            df,
            condition_order=condition_order,
            condition_col=self.CONDITION_COL,
            l0_col=l0_col,
            save_path=self.save_path / f"{prefix}_per_layer_by_condition.png"
        )

        plot_l0_per_layer_line(
            df,
            condition_order=condition_order,
            condition_col=self.CONDITION_COL,
            l0_col=l0_col,
            save_path=self.save_path / f"{prefix}_per_layer_line.png"
        )
