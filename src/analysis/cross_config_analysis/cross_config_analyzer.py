import os
from pathlib import Path
from typing import Dict, Type, Any

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
    CrossConfigEarlyLayerContributionStep
)
from src.analysis.cross_config_analysis.cross_config_error_ranking_step import CrossConfigErrorRankingStep
from src.analysis.cross_config_analysis.cross_config_feature_overlap_step import CrossConfigFeatureOverlapStep
from src.analysis.cross_config_analysis.cross_config_replacement_model_accuracy_step import (
    CrossConfigReplacementModelAccuracyStep
)
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import CrossConfigSubgraphFilterStep

STEP2CLASS: Dict[SupportedConfigAnalyzeStep, Type[CrossConfigAnalyzeStep]] = {
    SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION: CrossConfigEarlyLayerContributionStep,
    SupportedConfigAnalyzeStep.ERROR_RANKING: CrossConfigErrorRankingStep,
    SupportedConfigAnalyzeStep.FEATURE_OVERLAP: CrossConfigFeatureOverlapStep,
    SupportedConfigAnalyzeStep.REPLACEMENT_MODEL: CrossConfigReplacementModelAccuracyStep,
    SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: CrossConfigSubgraphFilterStep,
}


class CrossConfigAnalyzer:
    """Runs analysis steps across multiple configurations."""

    def __init__(self, config_results: Dict[SupportedConfigAnalyzeStep, Any], save_path: Path = None):
        """
        Initializes the cross-config analyzer with results from multiple configs.

        Args:
            config_results: Dictionary mapping config name to per-config results.
            save_path: Optional path for saving analysis outputs.
        """
        self.config_results = config_results
        self.save_path = save_path

    def run(self, **step_params) -> Dict[SupportedConfigAnalyzeStep, Any]:
        """
        Runs all registered cross-config analysis steps.

        Args:
            **step_params: Additional parameters passed to step constructors.

        Returns:
            Dictionary mapping SupportedConfigAnalyzeStep enum to results.
        """
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        all_results = {}
        for step_type, step_cls in STEP2CLASS.items():
            step: CrossConfigAnalyzeStep = step_cls(save_path=self.save_path, **step_params)
            step_result = step.run(self.config_results)
            if step_result is not None:
                all_results[step_type] = step_result
        return all_results
