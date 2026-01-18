from typing import Dict, List, Any

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
                 batch_size: int = None,
                 **kwargs):
        """
        Initializes the L0 replacement model step.

        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            prompt_ids: Optional list of prompt IDs to analyze.
            batch_size: If provided runs model with batched input.
        """
        self.batch_size = batch_size
        self.prompt_ids = prompt_ids
        self.model_manager = ReplacementModelManager(**get_method_kwargs(
            ReplacementModelManager.__init__, kwargs))
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> dict:
        """
        Run the L0 analysis on all graphs.

        Returns:
            Dictionary with 'results' (per-prompt L0 tensors) and 'd_transcoder'.
        """
        results = {}
        d_transcoder = None

        if self.batch_size:
            results = self.compute_l0_for_batches()
        else:
            for prompt_id, prompt in self.graph_analyzer.prompts.items():
                l0_per_layer = self.compute_l0_for_prompt(prompt)
                results[prompt_id] = l0_per_layer

        if not IS_TEST:
            d_transcoder = self.model_manager.get_d_transcoder()

        return {
            "results": results,
            "d_transcoder": d_transcoder
        }

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

    def compute_l0_for_batches(self) -> dict[str, Tensor]:
        """
        Compute L0 per layer for multiple prompts in batches.

        Returns:
            Dictionary mapping prompt_id to L0 tensor of shape (n_layers,).
        """
        prompt_ids = self.prompt_ids or list(self.graph_analyzer.prompts.keys())

        if IS_TEST:
            return {p_id: self._get_test_output() for p_id in prompt_ids}

        model = self.model_manager.get_model()
        prompts = [self.graph_analyzer.prompts[p_id] for p_id in prompt_ids]

        # Tokenize with padding, then ensure each is in correct format
        tokenized = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        token_list = [model.ensure_tokenized(tokenized["input_ids"][i]) for i in range(len(prompts))]
        prompt_tokens = torch.stack(token_list)

        l0_per_layer = self.compute_l0_per_layer(prompt_tokens)

        return {p_id: l0_per_layer[i] for i, p_id in enumerate(prompt_ids)}

    @staticmethod
    def _get_test_output(batch_size: int = None) -> Tensor:
        """
        Get dummy L0 output for testing without loading the model.

        Args:
            batch_size: If provided, returns test output for batched data.

        Returns:
            Tensor of dummy L0 values per layer.
        """
        dims = (ReplacementModelManager.DEFAULT_N_LAYERS,)
        if batch_size:
            dims = (batch_size, *dims)
        return torch.zeros(*dims, dtype=torch.float32)

    def compute_l0_per_layer(self, prompt_tokens: Tensor) -> Tensor:
        """
        Compute L0 (average active features) per layer.

        Args:
            prompt_tokens: Tokenized prompt tensor of shape (seq_len,) for single prompt,
                or (num_prompts, seq_len) for batched input.

        Returns:
            Tensor of shape (n_layers,) for single prompt,
            or (num_prompts, n_layers) for batched input.
        """
        features = self.model_manager.encode_features(prompt_tokens, self.batch_size)
        is_batched = prompt_tokens.dim() == 2

        non_zero = (features > 0).float()
        l0_per_token = non_zero.sum(dim=-1)  # sum over d_transcoder

        if is_batched:
            l0_per_layer = l0_per_token[:, :, 1:].mean(dim=-1)  # skip BOS
        else:
            l0_per_layer = l0_per_token[:, 1:].mean(dim=-1)  # skip BOS

        return l0_per_layer
