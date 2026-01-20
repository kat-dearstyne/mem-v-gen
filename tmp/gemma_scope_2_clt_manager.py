"""
Adapted L0 Calculation for Gemma Scope 2 CLTs

This integrates with your existing code structure (ReplacementModelManager pattern)
but uses the new Gemma Scope 2 CLTs from HuggingFace.

Key differences from your circuit_tracer approach:
1. Uses HuggingFace repos directly instead of circuit_tracer library
2. Works with Gemma 3 family (270m, 1b models have CLTs)
3. Can use SAELens for easier loading
"""

import os
from functools import partial
from typing import List, Optional

import torch
from torch import Tensor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json


class GemmaScope2CLTModelManager:
    """
    Replacement for ReplacementModelManager that works with Gemma Scope 2 CLTs.
    
    CLTs are available for Gemma 3 models (270m, 1b primarily).
    """
    
    # Base models corresponding to the CLT repos
    BASE_MODELS = {
        "270m-it": "google/gemma-3-270m-it",
        "1b-it": "google/gemma-3-1b-it",
        "270m-pt": "google/gemma-3-270m",
        "1b-pt": "google/gemma-3-1b",
    }
    
    CLT_REPOS = {
        "270m-pt": "google/gemma-scope-2-270m-pt",
        "270m-it": "google/gemma-scope-2-270m-it",
        "1b-pt": "google/gemma-scope-2-1b-pt",
        "1b-it": "google/gemma-scope-2-1b-it",
    }
    
    # Approximate model configurations
    MODEL_CONFIGS = {
        "270m": {"n_layers": 18, "d_model": 1536, "d_mlp": 6144},
        "1b": {"n_layers": 26, "d_model": 2048, "d_mlp": 8192},
    }
    
    def __init__(
        self,
        model_variant: str = "1b-it",
        clt_config: str = "width_262k_l0_medium",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the Gemma Scope 2 CLT manager.
        
        Args:
            model_variant: Model variant (e.g., "1b-it", "270m-pt")
            clt_config: CLT configuration from the repo
            device: Device to use
            dtype: Data type for weights
        """
        self.model_variant = model_variant
        self.clt_config = clt_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Extract model size from variant
        self.model_size = model_variant.split("-")[0]
        self.model_cfg = self.MODEL_CONFIGS.get(self.model_size, {})
        
        self._base_model = None
        self._tokenizer = None
        self._clt_weights = None
        self._clt_config = None
    
    @property
    def n_layers(self) -> int:
        return self.model_cfg.get("n_layers", 26)
    
    @property
    def d_model(self) -> int:
        return self.model_cfg.get("d_model", 2048)
    
    def load_base_model(self):
        """Load the base Gemma model and tokenizer."""
        if self._base_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            base_model_name = self.BASE_MODELS.get(self.model_variant)
            if not base_model_name:
                raise ValueError(f"Unknown model variant: {self.model_variant}")
            
            print(f"Loading base model: {base_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self._base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
            )
            print("Base model loaded.")
        
        return self._base_model, self._tokenizer
    
    def load_clt(self) -> tuple[dict, dict]:
        """Load CLT weights and configuration."""
        if self._clt_weights is None:
            repo_id = self.CLT_REPOS.get(self.model_variant)
            if not repo_id:
                raise ValueError(f"No CLT repo for variant: {self.model_variant}")

            clt_path = f"clt/{self.clt_config}"
            print(f"Loading CLT from {repo_id}/{clt_path}")

            # Try to load config
            try:
                config_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{clt_path}/config.json",
                )
                with open(config_file, "r") as f:
                    self._clt_config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load CLT config: {e}")
                self._clt_config = {}

            # Load weights - CLT weights are split per layer (keep on CPU to save GPU memory)
            self._clt_weights = {"layers": {}}
            for layer_idx in range(self.n_layers):
                weights_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{clt_path}/params_layer_{layer_idx}.safetensors",
                )
                layer_weights = load_file(weights_file, device="cpu")
                self._clt_weights["layers"][layer_idx] = layer_weights

                if layer_idx == 0:
                    print(f"Layer 0 weight keys: {list(layer_weights.keys())}")
                    for k, v in layer_weights.items():
                        print(f"  {k}: {v.shape}, dtype={v.dtype}")

            print(f"CLT loaded. Config: {self._clt_config}")
            print(f"Loaded {len(self._clt_weights['layers'])} layers")

        return self._clt_weights, self._clt_config
    
    def get_hidden_states(self, prompt: str) -> List[Tensor]:
        """
        Get hidden states from the base model for a given prompt.
        
        Args:
            prompt: Input text
            
        Returns:
            List of hidden states, one per layer
        """
        model, tokenizer = self.load_base_model()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # hidden_states is a tuple of (n_layers + 1) tensors
        # First one is embedding, rest are layer outputs
        return list(outputs.hidden_states)
    
    def get_hidden_states_batch(
        self, 
        prompts: List[str],
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """
        Get hidden states for multiple prompts.
        
        Args:
            prompts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape (num_prompts, n_layers+1, seq_len, d_model)
        """
        model, tokenizer = self.load_base_model()
        
        batch_size = batch_size or len(prompts)
        all_hidden_states = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Stack hidden states: (batch, n_layers+1, seq_len, d_model)
            batch_hs = torch.stack(outputs.hidden_states, dim=1)
            all_hidden_states.append(batch_hs)
        
        return torch.cat(all_hidden_states, dim=0)
    
    def encode_features(
        self,
        hidden_states: List[Tensor],
    ) -> Tensor:
        """
        Encode hidden states through the CLT to get feature activations.

        Args:
            hidden_states: List of hidden state tensors from the model

        Returns:
            Feature activations tensor
        """
        weights, config = self.load_clt()
        layer_weights = weights["layers"]
        threshold = config.get("threshold", 0.0)

        # Debug: print shapes
        print(f"Number of hidden states: {len(hidden_states)}")
        print(f"Hidden state 0 shape: {hidden_states[0].shape}")
        print(f"Config d_in: {config.get('d_in')}, d_sae: {config.get('d_sae')}")

        features_list = []

        # Skip embedding layer (index 0), use layers 1 to n_layers
        for layer_idx, hs in enumerate(hidden_states[1:]):  # Skip embedding
            if layer_idx not in layer_weights:
                continue

            lw = layer_weights[layer_idx]

            # Get encoder weights for this layer
            W_enc = None
            b_enc = None
            for enc_key in ["W_enc", "encoder.weight", "w_enc", "W_e"]:
                if enc_key in lw:
                    W_enc = lw[enc_key]
                    break
            for bias_key in ["b_enc", "encoder.bias", "b_e"]:
                if bias_key in lw:
                    b_enc = lw[bias_key]
                    break

            if W_enc is None:
                raise KeyError(f"Could not find encoder weights for layer {layer_idx}. Available: {list(lw.keys())}")

            # hs shape: (batch, seq_len, d_model) or (seq_len, d_model)
            is_single = hs.dim() == 2
            if is_single:
                hs = hs.unsqueeze(0)

            if layer_idx == 0:
                print(f"Layer 0 - hs shape: {hs.shape}, W_enc shape: {W_enc.shape}")

            # CLT W_enc might be (d_in, n_features) instead of (n_features, d_in)
            # Check and transpose if needed
            d_in = config.get("d_in", hs.shape[-1])

            # Move W_enc to GPU only when needed, compute, then free
            W_enc_gpu = W_enc.to(self.device)
            hs_gpu = hs.to(W_enc_gpu.dtype).to(self.device)

            if W_enc.shape[0] == d_in:
                pre_acts = torch.einsum("bsd,df->bsf", hs_gpu, W_enc_gpu)
            else:
                pre_acts = torch.einsum("bsd,fd->bsf", hs_gpu, W_enc_gpu)

            del W_enc_gpu, hs_gpu
            torch.cuda.empty_cache()

            if b_enc is not None:
                pre_acts = pre_acts + b_enc.to(self.device)

            # Apply activation (JumpReLU or ReLU)
            if threshold > 0:
                features = torch.where(
                    pre_acts > threshold,
                    pre_acts,
                    torch.zeros_like(pre_acts)
                )
            else:
                features = torch.relu(pre_acts)

            del pre_acts
            torch.cuda.empty_cache()

            if is_single:
                features = features.squeeze(0)

            # Compute L0 for this layer immediately and only keep that
            # This avoids storing the huge feature tensor
            l0_this_layer = (features > 0).float().sum(dim=-1)[:, 1:].mean(dim=-1) if features.dim() == 3 else (features > 0).float().sum(dim=-1)[1:].mean()
            features_list.append(l0_this_layer.cpu())

            del features
            torch.cuda.empty_cache()

        # Return L0 per layer: (n_layers,)
        return torch.stack(features_list, dim=0)
    
    def compute_l0_for_prompt(self, prompt: str) -> Tensor:
        """
        Compute L0 (number of active features) per layer for a single prompt.
        
        Args:
            prompt: Input text
            
        Returns:
            Tensor of L0 values per layer
        """
        hidden_states = self.get_hidden_states(prompt)
        features = self.encode_features(hidden_states)
        return self._compute_l0_from_features(features)
    
    def compute_l0_for_batch(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """
        Compute L0 per layer for multiple prompts.
        
        Args:
            prompts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape (num_prompts, n_layers)
        """
        hidden_states_batch = self.get_hidden_states_batch(prompts, batch_size)
        
        # Process each prompt's hidden states
        all_l0 = []
        for i in range(hidden_states_batch.shape[0]):
            # Get this prompt's hidden states as a list
            prompt_hs = [hidden_states_batch[i, layer] for layer in range(hidden_states_batch.shape[1])]
            features = self.encode_features(prompt_hs)
            l0 = self._compute_l0_from_features(features)
            all_l0.append(l0)
        
        return torch.stack(all_l0, dim=0)
    
    def _compute_l0_from_features(self, features: Tensor, skip_bos: bool = True) -> Tensor:
        """
        Compute L0 from feature activations.
        
        Args:
            features: Feature tensor of shape (n_layers, seq_len, n_features)
                      or (batch, n_layers, seq_len, n_features)
            skip_bos: Whether to skip the BOS token
            
        Returns:
            L0 per layer
        """
        # Count non-zero features
        non_zero = (features > 0).float()
        
        # Sum over feature dimension to get L0 per token
        l0_per_token = non_zero.sum(dim=-1)
        
        # Average over sequence (optionally skipping BOS)
        if skip_bos:
            if features.dim() == 3:
                l0_per_layer = l0_per_token[:, 1:].mean(dim=-1)
            else:
                l0_per_layer = l0_per_token[:, :, 1:].mean(dim=-1)
        else:
            l0_per_layer = l0_per_token.mean(dim=-1)
        
        return l0_per_layer
    
    def get_d_transcoder(self) -> int:
        """Get the number of features in the CLT."""
        weights, _ = self.load_clt()
        layer_weights = weights["layers"]

        # Get first layer's weights
        first_layer = layer_weights[0]
        for enc_key in ["W_enc", "encoder.weight", "w_enc", "W_e"]:
            if enc_key in first_layer:
                W = first_layer[enc_key]
                return W.shape[0]  # (n_features, d_model)

        raise KeyError("Could not determine d_transcoder from weights")


# =============================================================================
# Alternative: Using SAELens (if available)
# =============================================================================

def create_saelens_based_manager():
    """
    Alternative implementation using SAELens library.
    SAELens now supports CLT loading for Gemma Scope 2.
    """
    try:
        from sae_lens import SAE
        
        class SAELensBasedManager:
            def __init__(self, model_variant: str = "1b-it"):
                self.release = f"gemma-scope-2-{model_variant}-clt"
                self._clt = None
            
            def load_clt(self, sae_id: str = "width_262k_l0_medium"):
                if self._clt is None:
                    self._clt, self._cfg, self._sparsity = SAE.from_pretrained(
                        release=self.release,
                        sae_id=sae_id,
                    )
                return self._clt
            
            def encode(self, activations: Tensor) -> Tensor:
                clt = self.load_clt()
                return clt.encode(activations)
            
            def compute_l0(self, features: Tensor) -> Tensor:
                non_zero = (features > 0).float()
                return non_zero.sum(dim=-1).mean(dim=-1)
        
        return SAELensBasedManager
    
    except ImportError:
        print("SAELens not available. Use manual loading instead.")
        return None


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Gemma Scope 2 CLT L0 Calculator")
    print("="*60)
    
    # Example usage (requires actual model downloads)
    print("\nExample code to compute L0:")
    print("""
    # Initialize manager
    manager = GemmaScope2CLTModelManager(
        model_variant="1b-it",
        clt_config="width_262k_l0_medium",
    )
    
    # Compute L0 for a single prompt
    l0 = manager.compute_l0_for_prompt("Hello, how are you?")
    print(f"L0 per layer: {l0}")
    
    # Compute L0 for multiple prompts
    prompts = ["Hello world", "The quick brown fox"]
    l0_batch = manager.compute_l0_for_batch(prompts)
    print(f"L0 batch shape: {l0_batch.shape}")
    """)
    
    print("\nAvailable model variants:")
    for variant, repo in GemmaScope2CLTModelManager.CLT_REPOS.items():
        print(f"  {variant}: {repo}")
