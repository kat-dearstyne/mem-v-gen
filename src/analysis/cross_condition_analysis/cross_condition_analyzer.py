import os
from pathlib import Path
from typing import Dict, List, Type, Any, Optional

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_early_layer_step import (
    CrossConditionEarlyLayerStep
)
from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step
from src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step import (
    CrossConditionOverlapVisualizationStep
)
from src.analysis.cross_condition_analysis.cross_condition_feature_overlap_visualization_step import (
    CrossConditionFeatureOverlapVisualizationStep
)
from src.analysis.cross_condition_analysis.cross_condition_shared_features_visualization_step import (
    CrossConditionSharedFeaturesVisualizationStep
)


# Mapping of step names to step classes
CROSS_CONDITION_STEPS: List[Type[CrossConditionAnalyzeStep]] = [
    CrossConditionEarlyLayerStep,
    CrossConditionL0Step,
    CrossConditionOverlapVisualizationStep,
    CrossConditionFeatureOverlapVisualizationStep,
    CrossConditionSharedFeaturesVisualizationStep,
]


class CrossConditionAnalyzer:
    """
    Runs analysis steps comparing results across multiple conditions.

    Each condition represents the output of a CrossConfigAnalyzer run,
    allowing comparison of results from different experiment configurations,
    submodels, or parameter settings.
    """

    def __init__(self,
                 condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]],
                 save_path: Path = None):
        """
        Initializes the cross-condition analyzer.

        Args:
            condition_results: Dictionary mapping condition names to their
                CrossConfigAnalyzer results (Dict[SupportedConfigAnalyzeStep, Any]).
            save_path: Optional path for saving analysis outputs.
        """
        self.condition_results = condition_results
        self.save_path = save_path

    def run(self,
            condition_order: Optional[List[str]] = None,
            config_order: Optional[List[str]] = None,
            **step_params) -> Dict[str, Any]:
        """
        Runs all registered cross-condition analysis steps.

        Args:
            condition_order: Optional ordering for conditions in visualizations.
            config_order: Optional ordering for configs in visualizations.
            **step_params: Additional parameters passed to step constructors.

        Returns:
            Dictionary mapping step class names to their results.
        """
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        all_results = {}

        for step_cls in CROSS_CONDITION_STEPS:
            step: CrossConditionAnalyzeStep = step_cls(
                save_path=self.save_path,
                condition_order=condition_order,
                config_order=config_order,
                **step_params
            )
            step_result = step.run(self.condition_results)

            if step_result is not None:
                all_results[step_cls.__name__] = step_result

        if self.save_path:
            print(f"Saved cross-condition analysis results to: {self.save_path}")

        return all_results
