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
        features = self.model_manager.encode_features(prompt_tokens)
        return self.compute_l0_per_layer(features)

    def compute_l0_for_batches(self) -> dict[str, Tensor]:
        """
        Compute L0 per layer for multiple prompts in batches.

        Processes in batches to avoid memory issues, computing L0 for each
        batch and discarding features before processing the next batch.

        Returns:
            Dictionary mapping prompt_id to L0 tensor of shape (n_layers,).
        """
        prompt_ids = self.prompt_ids or list(self.graph_analyzer.prompts.keys())

        if IS_TEST:
            return {p_id: self._get_test_output() for p_id in prompt_ids}

        model = self.model_manager.get_model()
        prompts = [self.graph_analyzer.prompts[p_id] for p_id in prompt_ids]

        # Tokenize with padding
        tokenized = model.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        token_list = [model.ensure_tokenized(tokenized["input_ids"][i]) for i in range(len(prompts))]
        prompt_tokens = torch.stack(token_list)

        # Process in batches to save memory
        num_prompts = len(prompt_ids)
        all_l0 = []

        for batch_start in range(0, num_prompts, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_prompts)
            batch_tokens = prompt_tokens[batch_start:batch_end]

            features = self.model_manager.encode_features(batch_tokens, self.batch_size)
            batch_l0 = self.compute_l0_per_layer(features)
            all_l0.append(batch_l0.cpu())

            del features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        l0_per_layer = torch.cat(all_l0, dim=0)
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

    @staticmethod
    def compute_l0_per_layer(features: Tensor) -> Tensor:
        """
        Compute L0 (average active features) per layer from features tensor.

        Args:
            features: Features tensor of shape (n_layers, seq_len, d_transcoder) for single prompt,
                or (num_prompts, n_layers, seq_len, d_transcoder) for batched input.

        Returns:
            Tensor of shape (n_layers,) for single prompt,
            or (num_prompts, n_layers) for batched input.
        """
        is_batched = features.dim() == 4

        non_zero = (features > 0).float()
        l0_per_token = non_zero.sum(dim=-1)  # sum over d_transcoder

        if is_batched:
            l0_per_layer = l0_per_token[:, :, 1:].mean(dim=-1)  # skip BOS
        else:
            l0_per_layer = l0_per_token[:, 1:].mean(dim=-1)  # skip BOS

        return l0_per_layer
