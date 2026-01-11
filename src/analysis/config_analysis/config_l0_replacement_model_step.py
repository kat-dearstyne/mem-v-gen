from typing import List

import torch
from torch import Tensor

from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.constants import IS_TEST
from src.graph_analyzer import GraphAnalyzer
from src.replacement_model import ReplacementModelManager
from src.utils import get_method_kwargs


class ConfigL0ReplacementModelStep(ConfigAnalyzeStep):
    """
    Analysis step calculating L0 (number of active features) for each layer.
    """

    def __init__(self,
                 graph_analyzer: GraphAnalyzer,
                 prompt_ids: List[str] = None,
                 **kwargs):
        """
        Initializes the L0 replacement model step.

        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            prompt_ids: Optional list of prompt IDs to analyze.
        """
        self.prompt_ids = prompt_ids
        self.model_manager = ReplacementModelManager(**get_method_kwargs(
            ReplacementModelManager.__init__, kwargs))
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> dict:
        """
        Run the L0 analysis on all graphs.

        Returns:
            Dictionary mapping prompt IDs to L0 per layer tensors.
        """
        results = {}
        for prompt_id, graph in self.graph_analyzer.graphs.items():
            l0_per_layer = self.compute_l0_for_prompt(graph.prompt)
            results[prompt_id] = l0_per_layer
        return results

    def compute_l0_for_prompt(self, prompt_str: str) -> Tensor:
        """
        Compute L0 per layer for a single prompt.

        Args:
            prompt_str: The prompt string to analyze.

        Returns:
            Tensor of L0 values per layer.
        """
        if IS_TEST:
            return self._get_test_output()

        model = self.model_manager.get_model()
        prompt_tokens = model.ensure_tokenized(prompt_str)
        return self.compute_l0_per_layer(prompt_tokens)

    @staticmethod
    def _get_test_output() -> Tensor:
        """
        Get dummy L0 output for testing without loading the model.

        Returns:
            Tensor of dummy L0 values per layer.
        """
        return torch.zeros(ReplacementModelManager.DEFAULT_N_LAYERS, dtype=torch.float32)

    def compute_l0_per_layer(self, prompt_tokens: Tensor) -> Tensor:
        """
        Compute L0 (average active features) per layer.

        Args:
            prompt_tokens: Tokenized prompt tensor.

        Returns:
            Tensor of shape (n_layers,) with average L0 per layer.
        """
        features = self.model_manager.encode_features(prompt_tokens)

        non_zero = (features > 0).float()
        l0_per_token = non_zero.sum(dim=-1)
        l0_per_layer = l0_per_token[:, 1:].mean(dim=-1)  # skip BOS

        return l0_per_layer