from typing import List, Dict, Type

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.analysis.config_analysis.config_early_layer_contribution_step import ConfigEarlyLayerContributionStep
from src.analysis.config_analysis.config_error_ranking_step import ConfigErrorRankingStep
from src.analysis.config_analysis.config_feature_overlap_step import ConfigFeatureOverlapStep
from src.analysis.config_analysis.config_l0_replacement_model_step import ConfigL0ReplacementModelStep
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ConfigReplacementModelAccuracyStep
from src.analysis.config_analysis.config_subgraph_filter_step import ConfigSubgraphFilterStep
from src.analysis.config_analysis.config_token_subgraph_step import ConfigTokenSubgraphStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.graph_analyzer import GraphAnalyzer
from src.neuronpedia_manager import NeuronpediaManager

STEP2CLASS: Dict[SupportedConfigAnalyzeStep, Type[ConfigAnalyzeStep]] = {
    SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION: ConfigEarlyLayerContributionStep,
    SupportedConfigAnalyzeStep.ERROR_RANKING: ConfigErrorRankingStep,
    SupportedConfigAnalyzeStep.FEATURE_OVERLAP: ConfigFeatureOverlapStep,
    SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: ConfigL0ReplacementModelStep,
    SupportedConfigAnalyzeStep.REPLACEMENT_MODEL: ConfigReplacementModelAccuracyStep,
    SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: ConfigSubgraphFilterStep,
    SupportedConfigAnalyzeStep.TOKEN_SUBGRAPH: ConfigTokenSubgraphStep
}

class ConfigAnalyzer:
    """Runs analysis steps on a single configuration of graph pairs."""

    def __init__(self, neuronpedia_manager: NeuronpediaManager, prompts: Dict[str, str]):
        """
        Initializes the config analyzer with a GraphAnalyzer.

        Args:
            neuronpedia_manager: Manager for loading graphs from Neuronpedia.
            prompts: Dictionary mapping prompt ID to prompt string.
        """
        self.graph_analyzer = GraphAnalyzer(prompts=prompts, neuronpedia_manager=neuronpedia_manager)

    def run(self, analyze_steps: SupportedConfigAnalyzeStep | List[SupportedConfigAnalyzeStep],
            **step_params) -> Dict[SupportedConfigAnalyzeStep, any]:
        """
        Runs the specified analysis steps.

        Args:
            analyze_steps: Single step or list of steps to run.
            **step_params: Additional parameters passed to step constructors.

        Returns:
            Dictionary mapping SupportedConfigAnalyzeStep enum to results.
        """
        results = {}
        analyze_steps = [analyze_steps] if not isinstance(analyze_steps, list) else analyze_steps
        for step_type in analyze_steps:
            step_cls: Type[ConfigAnalyzeStep] | None = STEP2CLASS.get(step_type)
            assert step_cls is not None, f"UNKNOWN STEP: {step_type.name}"
            step: ConfigAnalyzeStep = step_cls(graph_analyzer=self.graph_analyzer, **step_params)
            results[step_type] = step.run()
        return results