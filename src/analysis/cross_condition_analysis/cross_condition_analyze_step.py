import abc
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CONFIG_NAME_COL, PROMPT_TYPE_COL
)


class CrossConditionAnalyzeStep(abc.ABC):
    """
    Base class for cross-condition analysis steps.

    Cross-condition steps compare results across multiple conditions,
    where each condition represents the output of a CrossConfigAnalyzer run.
    """

    # Subclasses should override these to specify which data to extract
    CONFIG_RESULTS_KEY: SupportedConfigAnalyzeStep = None
    RESULTS_SUB_KEY: Optional[str] = None  # For nested results (e.g., "intersection_metrics")

    def __init__(self, save_path: Path = None,
                 condition_order: Optional[List[str]] = None,
                 config_order: Optional[List[str]] = None,
                 **kwargs):
        """
        Initializes the cross-condition analysis step.

        Args:
            save_path: Base path for saving results.
            condition_order: Order for conditions in visualizations.
            config_order: Order for configs in visualizations.
            **kwargs: Additional parameters for the step.
        """
        self.save_path = save_path
        self.condition_order = condition_order
        self.config_order = config_order
        self.kwargs = kwargs

    @abc.abstractmethod
    def run(self, condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Any:
        """
        Runs the step logic.

        Args:
            condition_results: Dictionary mapping condition names to their
                CrossConfigAnalyzer results (Dict[SupportedConfigAnalyzeStep, Any]).

        Returns:
            Step-specific results.
        """

    def combine_condition_dataframes(
            self,
            condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]],
            add_condition_as_prompt_type: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Extracts and combines DataFrames from all conditions.

        Uses CONFIG_RESULTS_KEY and RESULTS_SUB_KEY to locate the DataFrame
        in each condition's results.

        Args:
            condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.
            add_condition_as_prompt_type: If True, adds condition name as prompt_type column.

        Returns:
            Combined DataFrame, or None if no data found.
        """
        if self.CONFIG_RESULTS_KEY is None:
            return None

        dfs = []

        for condition_name, step_results in condition_results.items():
            df = self._extract_dataframe(step_results)
            if df is None or df.empty:
                continue

            df = df.copy()

            # Add condition name as prompt_type if requested
            if add_condition_as_prompt_type:
                df[PROMPT_TYPE_COL] = condition_name

            dfs.append(df)

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)

        # Normalize config names (strip directory prefixes)
        if CONFIG_NAME_COL in combined.columns:
            combined[CONFIG_NAME_COL] = combined[CONFIG_NAME_COL].apply(
                lambda x: x.split('/')[-1] if '/' in str(x) else x
            )

        return combined

    def _extract_dataframe(
            self, step_results: Dict[SupportedConfigAnalyzeStep, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Extracts a DataFrame from step results using CONFIG_RESULTS_KEY and RESULTS_SUB_KEY.

        Args:
            step_results: Results dictionary from CrossConfigAnalyzer.

        Returns:
            Extracted DataFrame, or None if not found.
        """
        result = step_results.get(self.CONFIG_RESULTS_KEY)
        if result is None:
            return None

        # Handle nested results (e.g., {"intersection_metrics": df, "shared_features": df})
        if self.RESULTS_SUB_KEY is not None:
            if isinstance(result, dict):
                result = result.get(self.RESULTS_SUB_KEY)
            else:
                return None

        # Return DataFrame if valid
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result

        return None

    def get_ordering(self, df: pd.DataFrame) -> tuple[List[str], List[str]]:
        """
        Gets condition and config ordering, using provided values or deriving from data.

        Args:
            df: DataFrame with prompt_type and config_name columns.

        Returns:
            Tuple of (condition_order, config_order).
        """
        condition_order = self.condition_order
        config_order = self.config_order

        if condition_order is None and PROMPT_TYPE_COL in df.columns:
            condition_order = sorted(df[PROMPT_TYPE_COL].unique().tolist())

        if config_order is None and CONFIG_NAME_COL in df.columns:
            config_order = sorted(df[CONFIG_NAME_COL].unique().tolist())

        return condition_order or [], config_order or []
