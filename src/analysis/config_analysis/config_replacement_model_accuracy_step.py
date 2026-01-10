import inspect
import os
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import transformer_lens as tl
from circuit_tracer import ReplacementModel
from huggingface_hub import login
from torch import Tensor

from src.constants import IS_TEST, TOP_K
from src.analysis.config_analysis.config_analyze_step import ConfigAnalyzeStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.graph_manager import GraphManager
from src.metrics import ReplacementAccuracyMetrics
from src.utils import save_json

if TYPE_CHECKING:
    from src.graph_analyzer import GraphAnalyzer


ReplacementPredComparison = namedtuple("ReplacementPredComparison", [
    "prompt_tokens", "base_logits_BPV", "replacement_logits_BPV", "original_top_token_id"
])
ModelOutput = namedtuple("ModelOutput", ["logits_BPV", "top_token_id", "top_token"])


class ConfigReplacementModelAccuracyStep(ConfigAnalyzeStep):
    """
    Analysis step comparing base model and replacement model accuracy.

    Computes various metrics comparing outputs from the base model versus
    a replacement model that uses transcoder-reconstructed MLP activations.
    """
    HIGHER_IS_BETTER_METRICS = {ReplacementAccuracyMetrics.LAST_TOKEN_COSINE,
                                ReplacementAccuracyMetrics.CUMULATIVE_COSINE,
                                ReplacementAccuracyMetrics.ORIGINAL_ACCURACY,
                                ReplacementAccuracyMetrics.TOP_K_AGREEMENT,
                                ReplacementAccuracyMetrics.REPLACEMENT_PROB_OF_ORIGINAL_TOP}
    LOWER_IS_BETTER_METRICS = {ReplacementAccuracyMetrics.KL_DIVERGENCE}
    BASE_MODEL = "google/gemma-2-2b"
    SUB_MODEL = "mntss/clt-gemma-2-2b-426k"

    def __init__(self,
                 graph_analyzer: "GraphAnalyzer",
                 memorized_completion: str,
                 metrics2run: set[ReplacementAccuracyMetrics] | None = None,
                 **kwargs):
        """
        Initializes the config replacement model accuracy step.

        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            memorized_completion: Expected memorized completion for prompt.
            metrics2run: Set of metrics to compute (default: all metrics).
        """
        self.metrics2run = {e for e in ReplacementAccuracyMetrics} if not metrics2run else metrics2run
        self.memorized_completion = memorized_completion
        self.model = None
        self.metric2func = {
            ReplacementAccuracyMetrics.PER_POSITION_COSINE: self.get_per_position_cosine,
            ReplacementAccuracyMetrics.PER_POSITION_KL: self.get_per_position_kl_divergence,
            ReplacementAccuracyMetrics.PER_POSITION_ARGMAX_MATCH: self.get_per_position_argmax_match,
            ReplacementAccuracyMetrics.LAST_TOKEN_COSINE: self.get_last_token_accuracy,
            ReplacementAccuracyMetrics.ORIGINAL_ACCURACY: self.get_original_accuracy_metric,
            ReplacementAccuracyMetrics.KL_DIVERGENCE: self.get_kl_divergence,
            ReplacementAccuracyMetrics.TOP_K_AGREEMENT: self.get_top_k_agreement,
            ReplacementAccuracyMetrics.REPLACEMENT_PROB_OF_ORIGINAL_TOP: self.get_replacement_prob_of_original_top
        }
        super().__init__(graph_analyzer=graph_analyzer, **kwargs)

    def run(self) -> dict:
        """
        Run the analysis step on all graphs.

        Returns:
            Dictionary mapping prompt IDs to their metric results.
        """
        print("Loading model...")
        model = self.load_model()
        print("Model loaded.\n")

        results = {}
        for prompt_id, graph in self.graph_analyzer.graphs.items():
            metrics = self.run_accuracy_test(graph.prompt)
            results[prompt_id] = metrics

        return results

    def get_model(self) -> ReplacementModel | None:
        """
        Get the model instance, loading it if necessary.

        Returns:
            The ReplacementModel instance, or None in test mode.
        """
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def load_model(self) -> ReplacementModel | None:
        """
        Load the ReplacementModel with CLT transcoders.

        Returns:
            The loaded ReplacementModel, or None in test mode.
        """
        hf_token = os.environ.get("HF_TOKEN")

        if IS_TEST:
            return None

        if hf_token:
            login(hf_token)
        model = ReplacementModel.from_pretrained(
            self.BASE_MODEL,
            self.SUB_MODEL,
            dtype=torch.bfloat16,
        )
        return model

    def run_accuracy_test(self, prompt_str: str) -> dict[str, Any]:
        """
        Run accuracy comparison for a single prompt.

        Args:
            prompt_str: The prompt string to test.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        results = {}
        if IS_TEST:
            # Dummy tensors for testing without model
            prompt_tokens, base_output, replacement_output = self._get_test_output(results=results)
        else:
            prompt_tokens, base_output, replacement_output = self._get_model_outputs(prompt_str, results=results)

        # Per-position metrics

        params = dict(base_logits_BPV=base_output.logits_BPV,
                      replacement_logits_BPV=replacement_output.logits_BPV)
        for metric in ReplacementAccuracyMetrics:
            if metric in self.metrics2run and metric in self.metric2func:
                func = self.metric2func.get(metric)
                include_tokens = "prompt_tokens" in inspect.signature(func).parameters
                result = func(prompt_tokens=prompt_tokens, **params) if include_tokens else func(**params)
                results[metric] = result

        if per_pos_cosine := results.get(ReplacementAccuracyMetrics.PER_POSITION_COSINE):
            cumulative_acc = sum(per_pos_cosine) / len(per_pos_cosine)
            results[ReplacementAccuracyMetrics.CUMULATIVE_COSINE] = cumulative_acc

        results = {
            ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN: base_output.top_token,
            ReplacementAccuracyMetrics.REPLACEMENT_TOP_TOKEN: replacement_output.top_token,
            **results
        }
        return results

    @staticmethod
    def save_results(results: dict, output_dir: Path) -> None:
        """
        Save results to JSON file.

        Args:
            results: Dictionary structured as {config_name: {SupportedConfigAnalyzeStep: {condition: metrics}}}.
            output_dir: Directory to save the output file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "error_hypothesis_analysis.json"

        # Convert namedtuples to dicts for JSON serialization
        serializable_results = {}
        for config_name, config_results in results.items():
            condition_metrics = config_results[SupportedConfigAnalyzeStep.REPLACEMENT_MODEL]
            serializable_results[config_name] = {
                condition: metrics
                for condition, metrics in condition_metrics.items()
            }

        save_json(serializable_results, output_path)

        print(f"\nResults saved to: {output_path}")

    def _get_model_outputs(self, prompt_str: str,
                           results: dict[ReplacementAccuracyMetrics, Any]
                           ) -> tuple[Tensor, ModelOutput, ModelOutput]:
        """
        Get base and replacement model outputs for a prompt.

        Args:
            prompt_str: The prompt string to process.
            results: Dictionary to store perplexity metrics (modified in place).

        Returns:
            Tuple of (prompt_tokens, base_output, replacement_output).
        """
        model = self.get_model()
        prompt_tokens = model.ensure_tokenized(prompt_str)

        with torch.no_grad():
            base_logits_BPV = model(prompt_tokens, return_type='logits')
            replacement_logits_BPV = self.get_replacement_logits(prompt_tokens)

        base_output = self._extract_model_output(base_logits_BPV)
        replacement_output = self._extract_model_output(replacement_logits_BPV)

        if ReplacementAccuracyMetrics.PERPLEXITY_FULL in self.metrics2run or \
                ReplacementAccuracyMetrics.PERPLEXITY_LAST_TOKEN in self.metrics2run:
            ppl_last, ppl_full = self.get_base_perplexity(base_logits_BPV, prompt_tokens)
            results[ReplacementAccuracyMetrics.PERPLEXITY_FULL] = ppl_full
            results[ReplacementAccuracyMetrics.PERPLEXITY_LAST_TOKEN] = ppl_last

        if ReplacementAccuracyMetrics.PER_POSITION_CROSS_ENTROPY in self.metrics2run:
            per_pos_ce = self.get_per_position_cross_entropy(base_logits_BPV, prompt_tokens)
            results[ReplacementAccuracyMetrics.PERPLEXITY_FULL] = per_pos_ce

        return prompt_tokens, base_output, replacement_output

    def _get_test_output(self, results: dict[ReplacementAccuracyMetrics, Any]
                         ) -> tuple[Tensor, ModelOutput, ModelOutput]:
        """
        Get dummy outputs for testing without loading the model.

        Args:
            results: Dictionary to store placeholder metrics (modified in place).

        Returns:
            Tuple of (prompt_tokens, base_output, replacement_output) with dummy data.
        """
        prompt_tokens = torch.zeros((1, 10), dtype=torch.long)
        base_output = ModelOutput(logits_BPV=torch.randn((1, 10, 256000), dtype=torch.bfloat16),
                                  top_token_id=torch.randn((1, 10, 256000), dtype=torch.bfloat16),
                                  top_token="<test>")
        replacement_output = deepcopy(base_output)
        if ReplacementAccuracyMetrics.PERPLEXITY_FULL in self.metrics2run or \
                ReplacementAccuracyMetrics.PERPLEXITY_LAST_TOKEN in self.metrics2run:
            results[ReplacementAccuracyMetrics.PERPLEXITY_FULL] = None
            results[ReplacementAccuracyMetrics.PERPLEXITY_LAST_TOKEN] = None

        if ReplacementAccuracyMetrics.PER_POSITION_CROSS_ENTROPY in self.metrics2run:
            results[ReplacementAccuracyMetrics.PERPLEXITY_FULL] = None
        return prompt_tokens, base_output, replacement_output

    def get_replacement_logits(self, prompt_tokens: Tensor) -> Tensor:
        """
        Get logits using transcoder-reconstructed MLP activations.

        Hooks into each layer to encode activations through transcoders,
        then decode them back, replacing the original MLP outputs.

        Args:
            prompt_tokens: Tokenized prompt tensor.

        Returns:
            Logits tensor from the replacement model.
        """
        model = self.get_model()
        features = torch.zeros(
            (model.cfg.n_layers, len(prompt_tokens), model.transcoders.d_transcoder),
            dtype=torch.bfloat16
        ).to(model.cfg.device)
        bos_actv = torch.zeros((model.cfg.d_model), dtype=torch.bfloat16).to(model.cfg.device)

        def input_hook_fn(value, hook, layer):
            nonlocal bos_actv
            bos_actv = value[0, 0]
            features[layer] = model.transcoders.encode_layer(value, layer)
            features[:, 0] = 0.  # exclude bos
            return value

        def output_hook_fn(value, hook, layer):
            mlp_outs = model.transcoders.decode(features)
            mlp_out = mlp_outs[layer].unsqueeze(0)
            mlp_out[:, 0] = bos_actv
            return mlp_out

        all_hooks = []
        for layer in range(model.cfg.n_layers):
            input_hook_fn_partial = partial(input_hook_fn, layer=layer)
            all_hooks.append((tl.utils.get_act_name(model.feature_input_hook[5:], layer), input_hook_fn_partial))

            output_hook_fn_partial = partial(output_hook_fn, layer=layer)
            all_hooks.append((tl.utils.get_act_name(model.feature_output_hook[5:], layer), output_hook_fn_partial))

        with torch.no_grad():
            logits = model.run_with_hooks(prompt_tokens, fwd_hooks=all_hooks, return_type='logits')
        return logits

    def get_base_perplexity(self, base_logits_BPV: Tensor,
                            prompt_tokens: Tensor) -> tuple[float | None, float | None]:
        """
        Calculate perplexity for the prompt and the first token of the memorized completion.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            prompt_tokens: Tokenized prompt tensor.

        Returns:
            Tuple of (last_token_ppl, full_ppl), or (None, None) if computation fails.
        """
        ce_per_pos = self.get_per_position_cross_entropy(base_logits_BPV, prompt_tokens)
        if not ce_per_pos:
            return None, None

        last_token_ppl = np.exp(ce_per_pos[-1])
        full_ppl = np.exp(np.mean(ce_per_pos))

        return last_token_ppl, full_ppl

    def get_per_position_cross_entropy(self, base_logits_BPV: Tensor,
                                       prompt_tokens: Tensor) -> list[float]:
        """
        Compute cross-entropy at each position (position i predicting token i+1).

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            prompt_tokens: Tokenized prompt, either (seq_len,) or (1, seq_len).

        Returns:
            List of cross-entropy values, one per position.
        """
        model = self.get_model()

        # Tokenize the completion to get the first token
        completion_tokens = model.tokenizer.encode(self.memorized_completion, add_special_tokens=False)
        first_completion_token = completion_tokens[0]

        # Ensure prompt_tokens is 1D
        tokens = prompt_tokens.squeeze() if prompt_tokens.dim() > 1 else prompt_tokens

        logits = base_logits_BPV[0].float()  # (seq_len, vocab)
        # Targets: tokens[1:] for prefix, then first_completion_token for last position
        completion_tensor = torch.tensor([first_completion_token], dtype=torch.long, device=tokens.device)
        targets = torch.cat([tokens[1:].long(), completion_tensor])

        # Sanity check lengths
        assert logits.shape[0] == targets.shape[
            0], f"Length mismatch: logits {logits.shape[0]} vs targets {targets.shape[0]}"

        ce_per_pos = []
        for i in range(logits.shape[0]):
            ce = torch.nn.functional.cross_entropy(logits[i].unsqueeze(0), targets[i:i + 1])
            ce_per_pos.append(ce.item())

        return ce_per_pos

    @staticmethod
    def get_per_position_cosine(base_logits_BPV: Tensor,
                                replacement_logits_BPV: Tensor) -> list[float]:
        """
        Compute cosine similarity at each token position.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            List of cosine similarities, one per position.
        """
        num_tokens = base_logits_BPV.shape[1]
        cosines = []

        for i in range(num_tokens):
            base_logits_V = base_logits_BPV[0, i]
            replacement_logits_V = replacement_logits_BPV[0, i]

            base_norm = torch.linalg.norm(base_logits_V)
            replacement_norm = torch.linalg.norm(replacement_logits_V)

            cosine = (base_logits_V.T @ replacement_logits_V.T) / (base_norm * replacement_norm)
            cosines.append(torch.abs(cosine).item())

        return cosines

    @staticmethod
    def get_per_position_kl_divergence(base_logits_BPV: Tensor,
                                       replacement_logits_BPV: Tensor) -> list[float]:
        """
        Compute KL divergence at each token position.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            List of KL divergences (in nats), one per position.
        """
        num_tokens = base_logits_BPV.shape[1]
        kl_divs = []

        for i in range(num_tokens):
            base_logits = base_logits_BPV[0, i].float()
            replacement_logits = replacement_logits_BPV[0, i].float()

            base_log_probs = torch.log_softmax(base_logits, dim=-1)
            replacement_log_probs = torch.log_softmax(replacement_logits, dim=-1)
            base_probs = base_log_probs.exp()

            kl_div = torch.sum(base_probs * (base_log_probs - replacement_log_probs))
            kl_divs.append(kl_div.item())

        return kl_divs

    @staticmethod
    def get_last_token_accuracy(base_logits_BPV: Tensor,
                                replacement_logits_BPV: Tensor) -> float:
        """
        Compute accuracy as cosine similarity between logit vectors at last position.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            Cosine similarity between final position logits.
        """
        base_logits_V = base_logits_BPV[0, -1]
        replacement_logits_V = replacement_logits_BPV[0, -1]

        base_norm = torch.linalg.norm(base_logits_V)
        replacement_norm = torch.linalg.norm(replacement_logits_V)

        accuracy = (base_logits_V.T @ replacement_logits_V.T) / (base_norm * replacement_norm)
        accuracy = torch.abs(accuracy).item()

        return accuracy

    @staticmethod
    def get_per_position_argmax_match(base_logits_BPV: Tensor,
                                      replacement_logits_BPV: Tensor) -> list[int]:
        """
        Compute argmax match at each token position.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            List of matches (1 if argmax matches, 0 otherwise), one per position.
        """
        base_argmax = base_logits_BPV[0].argmax(dim=-1)  # (seq_len,)
        replacement_argmax = replacement_logits_BPV[0].argmax(dim=-1)  # (seq_len,)
        matches = (base_argmax == replacement_argmax).int().tolist()
        return matches

    @staticmethod
    def get_original_accuracy_metric(base_logits_BPV: Tensor,
                                     replacement_logits_BPV: Tensor,
                                     prompt_tokens: Tensor) -> float:
        """
        Compute accuracy by directly comparing argmax predictions.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).
            prompt_tokens: Tokenized prompt tensor.

        Returns:
            Fraction of positions where argmax predictions match.
        """
        base_argmax = base_logits_BPV.argmax(dim=-1)
        replacement_argmax = replacement_logits_BPV.argmax(dim=-1)

        repl_acc = (base_argmax == replacement_argmax).sum() / prompt_tokens.numel()
        return repl_acc.item()

    @staticmethod
    def get_kl_divergence(base_logits_BPV: Tensor,
                          replacement_logits_BPV: Tensor) -> float:
        """
        Compute KL divergence between probability distributions at the last position.

        KL(base || replacement) measures how much information is lost when using
        the replacement distribution to approximate the base distribution.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            KL divergence (in nats) at the final position.
        """
        base_logits = base_logits_BPV[0, -1].float()
        replacement_logits = replacement_logits_BPV[0, -1].float()

        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        replacement_log_probs = torch.log_softmax(replacement_logits, dim=-1)
        base_probs = base_log_probs.exp()

        kl_div = torch.sum(
            base_probs * (base_log_probs - replacement_log_probs)
        )
        return kl_div.item()

    @staticmethod
    def get_top_k_agreement(base_logits_BPV: Tensor,
                            replacement_logits_BPV: Tensor,
                            k: int = TOP_K) -> float:
        """
        Compute the fraction of top-k tokens that overlap between distributions.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).
            k: Number of top tokens to compare.

        Returns:
            Fraction of top-k tokens that appear in both distributions (0 to 1).
        """
        base_logits = base_logits_BPV[0, -1]
        replacement_logits = replacement_logits_BPV[0, -1]

        base_top_k = torch.topk(base_logits, k).indices.tolist()
        replacement_top_k = torch.topk(replacement_logits, k).indices.tolist()

        overlap = len(set(base_top_k) & set(replacement_top_k))
        return overlap / k

    @staticmethod
    def get_replacement_prob_of_original_top(base_logits_BPV: Tensor,
                                             replacement_logits_BPV: Tensor) -> float:
        """
        Get the probability the replacement model assigns to the original model's top prediction.

        Args:
            base_logits_BPV: Base model logits (batch, position, vocab).
            replacement_logits_BPV: Replacement model logits (batch, position, vocab).

        Returns:
            Probability (0 to 1) that replacement model assigns to original's top token.
        """
        base_logits = base_logits_BPV[0, -1]
        replacement_logits = replacement_logits_BPV[0, -1].float()

        original_top_token_id = base_logits.argmax().item()
        replacement_probs = torch.softmax(replacement_logits, dim=-1)

        return replacement_probs[original_top_token_id].item()

    def _extract_model_output(self, logits_bpv: Tensor) -> ModelOutput:
        """
        Extract model output from logits tensor.

        Args:
            logits_bpv: Logits tensor (batch, position, vocab).

        Returns:
            ModelOutput with logits, top token ID, and decoded top token.
        """
        top_token_id = logits_bpv[0, -1].argmax().item()
        output = ModelOutput(logits_BPV=logits_bpv,
                             top_token_id=top_token_id,
                             top_token=self.get_model().tokenizer.decode([top_token_id]))
        return output
