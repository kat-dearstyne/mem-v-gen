from typing import List, Dict, Type

from src.analysis.config_analysis.config_error_ranking_step import ConfigErrorRankingStep
from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ConfigReplacementModelAccuracyStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.graph_manager import GraphManager
from src.neuronpedia_manager import NeuronpediaManager

STEP2CLASS: Dict[SupportedConfigAnalyzeStep, Type[ConfigAnalyzeStep]] = {
    SupportedConfigAnalyzeStep.ERROR_RANKING: ConfigErrorRankingStep,
    SupportedConfigAnalyzeStep.REPLACEMENT_MODEL: ConfigReplacementModelAccuracyStep
}

class ConfigAnalyzer:
    """Runs analysis steps on a single configuration of graph pairs."""

    def __init__(self, neuronpedia_manager: NeuronpediaManager = None,
                 prompts: Dict[str, str] = None,
                 graph_managers: Dict[str, GraphManager] = None):
        """
        Initializes the config analyzer with graphs or loading parameters.

        Args:
            neuronpedia_manager: Manager for loading graphs from Neuronpedia.
            prompts: Dictionary mapping prompt type to prompt string.
            graph_managers: Pre-loaded graph managers (alternative to loading).
        """
        assert (prompts and neuronpedia_manager) or graph_managers, ("Must provide either neuronpedia manager & prompts "
                                                                     "for loading graphs or the graphs themselves")
        self.neuronpedia_manager = neuronpedia_manager
        self.prompts = prompts
        self.graph_managers = graph_managers

        if not self.graph_managers:
            self.graph_managers = {prompt_type: self.neuronpedia_manager.create_or_load_graph(prompt)
                                    for prompt_type, prompt in prompts.items()}
            self.prompts = {prompt_type: graph for prompt_type, graph in self.graph_managers.items()}

    def run(self, analyze_steps: SupportedConfigAnalyzeStep | List[SupportedConfigAnalyzeStep],
            conditions: List[str] = None,
            **step_params):
        """
        Runs the specified analysis steps on the graphs.

        Args:
            analyze_steps: Single step or list of steps to run.
            conditions: Optional list of condition names to filter graphs.
            **step_params: Additional parameters passed to step constructors.

        Returns:
            Dictionary mapping step name to results.
        """
        results = {}
        analyze_steps = [analyze_steps] if not isinstance(analyze_steps, list) else analyze_steps
        for step_type in analyze_steps:
            step_cls: Type[ConfigAnalyzeStep] | None = STEP2CLASS.get(step_type)
            assert step_cls is not None, f"UNKNOWN STEP: {step_type.name}"
            step: ConfigAnalyzeStep = step_cls(**step_params)
            if conditions:
                graphs = [self.graph_managers.get(condition) for condition in conditions]
                assert all(graphs), f"Unknown conditions: {set(self.prompts.keys()).difference(conditions)}"
            else:
                graphs = list(self.graph_managers.values())
            results[step_type.name] = step.run(graphs, conditions)
        return results