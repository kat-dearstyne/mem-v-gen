import os
from functools import partial

import torch
import transformer_lens as tl
from circuit_tracer import ReplacementModel
from huggingface_hub import login
from torch import Tensor

from src.constants import IS_TEST

class SubModelType:
    PLT = 0
    CLT = 1

class ReplacementModelManager:
    DEFAULT_BASE_MODEL = "google/gemma-2-2b"
    DEFAULT_N_LAYERS = 26
    DEFAULT_SUB_MODELS = {
        SubModelType.PLT: "mntss/gemma-scope-transcoders",
        SubModelType.CLT:  "mntss/clt-gemma-2-2b-426k"
    }
    def __init__(self, base_model: str = DEFAULT_BASE_MODEL, sub_model: str | int = SubModelType.CLT):
        """
        Initializes replacement model with the base model and submodel.
        Args:
            base_model: The base model.
            sub_model: Used as replacement model. May be model name or CLT/PLT to use default.
        """
        self.base_model = base_model
        # in case PLT or CLT is given instead of model name
        sub_model = self.DEFAULT_SUB_MODELS.get(sub_model, sub_model)
        self.sub_model = sub_model
        self.__model = None

    def get_model(self) -> ReplacementModel | None:
        """
        Get the model instance, loading it if necessary.

        Returns:
            The ReplacementModel instance, or None in test mode.
        """
        if self.__model is None:
            print("Loading model...")
            self.__model = self.load_model()
            print("Model loaded.\n")
        return self.__model

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
        print(f"Loading submodel: {self.sub_model}")
        model = ReplacementModel.from_pretrained(
            self.base_model,
            self.sub_model,
            dtype=torch.bfloat16,
        )
        return model

    def encode_features(self, prompt_tokens: Tensor) -> Tensor:
        """
        Encode activations through transcoders for all layers.

        Args:
            prompt_tokens: Tokenized prompt tensor.

        Returns:
            Features tensor of shape (n_layers, seq_len, d_transcoder).
        """
        model = self.get_model()
        features = torch.zeros(
            (model.cfg.n_layers, len(prompt_tokens), model.transcoders.d_transcoder),
            dtype=torch.bfloat16
        ).to(model.cfg.device)

        def input_hook_fn(value, hook, layer):
            features[layer] = model.transcoders.encode_layer(value, layer)
            features[:, 0] = 0.  # exclude bos
            return value

        all_hooks = []
        for layer in range(model.cfg.n_layers):
            input_hook_fn_partial = partial(input_hook_fn, layer=layer)
            hook_name = self._get_hook_name(model, 'feature_input_hook', layer)
            all_hooks.append((hook_name, input_hook_fn_partial))

        with torch.no_grad():
            model.run_with_hooks(prompt_tokens, fwd_hooks=all_hooks)

        return features

    def get_replacement_logits(self, prompt_tokens: Tensor) -> Tensor:
        """
        Get logits using transcoder-reconstructed MLP activations.

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
            input_hook_name = self._get_hook_name(model, 'feature_input_hook', layer)
            all_hooks.append((input_hook_name, input_hook_fn_partial))

            output_hook_fn_partial = partial(output_hook_fn, layer=layer)
            output_hook_name = self._get_hook_name(model, 'feature_output_hook', layer)
            all_hooks.append((output_hook_name, output_hook_fn_partial))

        with torch.no_grad():
            logits = model.run_with_hooks(prompt_tokens, fwd_hooks=all_hooks, return_type='logits')
        return logits

    def get_d_transcoder(self) -> int:
        """
        Get the number of features in the transcoder.

        Returns:
            Number of transcoder features (d_transcoder).
        """
        model = self.get_model()
        return model.transcoders.d_transcoder

    @staticmethod
    def _get_hook_name(model, hook_attr: str, layer: int) -> str:
        """
        Get the full hook name for a given layer.

        Args:
            model: The ReplacementModel instance.
            hook_attr: Name of the hook attribute (e.g., 'feature_input_hook').
            layer: Layer index.

        Returns:
            Full hook name string (e.g., 'blocks.0.hook_resid_mid').
        """
        hook_point = getattr(model.transcoders, hook_attr)
        return f"blocks.{layer}.{hook_point}"