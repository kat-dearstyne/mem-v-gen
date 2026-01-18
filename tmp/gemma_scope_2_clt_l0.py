"""
Gemma Scope 2 CLT L0 Calculation

This script demonstrates how to load Cross-Layer Transcoders (CLTs) from Gemma Scope 2
and calculate L0 norm (number of active features).

CLTs are available for:
- google/gemma-scope-2-270m-pt (and -it)
- google/gemma-scope-2-1b-pt (and -it)

Note: Larger models (4B, 12B, 27B) also have `clt` folders but CLTs were primarily
released for 270m and 1b models according to the announcement.
"""

import torch
from typing import Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import json


# =============================================================================
# Option 1: Using SAELens (Recommended if it works for your use case)
# =============================================================================

def load_clt_with_saelens():
    """
    Load CLT using SAELens library.
    SAELens added CLT support in recent versions.
    
    pip install sae-lens
    """
    try:
        from sae_lens import SAE
        
        # Example: Load CLT for Gemma 3 1B IT
        # Note: Check available sae_ids by exploring the HuggingFace repo
        clt, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2-1b-it-clt",  # or gemma-scope-2-270m-it-clt
            sae_id="width_262k_l0_medium",  # adjust based on available configs
        )
        return clt, cfg_dict
    except Exception as e:
        print(f"SAELens loading failed: {e}")
        print("Falling back to manual loading...")
        return None, None


# =============================================================================
# Option 2: Manual Loading (More flexible, works without SAELens)
# =============================================================================

class CrossLayerTranscoder(torch.nn.Module):
    """
    Cross-Layer Transcoder implementation for Gemma Scope 2.
    
    CLTs take residual stream values just before each MLP layer and
    reconstruct the whole model's MLP outputs.
    
    Architecture:
    - Encoder: Maps input activations to sparse feature space
    - Decoder: Maps features back to MLP output space (for all layers)
    """
    
    def __init__(
        self,
        d_model: int,
        n_features: int,
        n_layers: int,
        threshold: float = 0.0,
        has_skip: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.n_layers = n_layers
        self.threshold = threshold
        self.has_skip = has_skip
        self.dtype = dtype
        
        # Encoder weights: (n_layers, n_features, d_model)
        # Each layer has its own encoder
        self.W_enc = torch.nn.Parameter(
            torch.zeros(n_layers, n_features, d_model, dtype=dtype)
        )
        self.b_enc = torch.nn.Parameter(
            torch.zeros(n_layers, n_features, dtype=dtype)
        )
        
        # Decoder weights: Maps features to MLP outputs for downstream layers
        # Shape depends on whether it's truly cross-layer or per-layer
        # For CLT: (n_features, n_layers, d_model) - each feature affects all downstream layers
        self.W_dec = torch.nn.Parameter(
            torch.zeros(n_features, n_layers, d_model, dtype=dtype)
        )
        
        # Optional: Skip connection scaling
        if has_skip:
            self.skip_scale = torch.nn.Parameter(
                torch.ones(n_layers, dtype=dtype)
            )
        
        # JumpReLU threshold (if applicable)
        if threshold > 0:
            self.register_buffer(
                "threshold_tensor",
                torch.tensor(threshold, dtype=dtype)
            )
    
    def encode(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Encode activations to feature space for a specific layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            layer: Layer index
            
        Returns:
            Feature activations of shape (batch, seq_len, n_features)
        """
        # Linear transformation
        pre_acts = torch.einsum("bsd,fd->bsf", x, self.W_enc[layer]) + self.b_enc[layer]
        
        # Apply activation (JumpReLU if threshold > 0, else ReLU)
        if self.threshold > 0:
            acts = torch.where(
                pre_acts > self.threshold,
                pre_acts,
                torch.zeros_like(pre_acts)
            )
        else:
            acts = torch.relu(pre_acts)
        
        return acts
    
    def encode_all_layers(
        self, 
        residual_streams: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode residual streams from all layers.
        
        Args:
            residual_streams: List of tensors, one per layer, 
                              each of shape (batch, seq_len, d_model)
                              
        Returns:
            Feature activations of shape (batch, n_layers, seq_len, n_features)
        """
        batch_size = residual_streams[0].shape[0]
        seq_len = residual_streams[0].shape[1]
        
        features = torch.zeros(
            batch_size, self.n_layers, seq_len, self.n_features,
            dtype=self.dtype,
            device=residual_streams[0].device
        )
        
        for layer in range(self.n_layers):
            features[:, layer] = self.encode(residual_streams[layer], layer)
        
        return features
    
    def compute_l0(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute L0 norm (count of active features) per layer.
        
        Args:
            features: Tensor of shape (batch, n_layers, seq_len, n_features)
                      or (n_layers, seq_len, n_features) for single example
                      
        Returns:
            L0 per layer, shape (batch, n_layers) or (n_layers,)
        """
        is_single = features.dim() == 3
        if is_single:
            features = features.unsqueeze(0)
        
        # Count non-zero features
        non_zero = (features > 0).float()
        
        # Sum over features dimension, average over sequence (excluding BOS)
        l0_per_token = non_zero.sum(dim=-1)  # (batch, n_layers, seq_len)
        l0_per_layer = l0_per_token[:, :, 1:].mean(dim=-1)  # Skip BOS token
        
        if is_single:
            return l0_per_layer.squeeze(0)
        return l0_per_layer


def load_clt_from_huggingface(
    repo_id: str = "google/gemma-scope-2-1b-it",
    clt_path: str = "clt/width_262k_l0_medium",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[dict, dict]:
    """
    Manually load CLT weights from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID
        clt_path: Path to CLT within the repo
        device: Device to load to
        
    Returns:
        Tuple of (weights dict, config dict)
    """
    # Download config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{clt_path}/config.json",
        )
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Could not load config: {e}")
        config = {}
    
    # Download weights
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{clt_path}/params.safetensors",
    )
    weights = load_file(weights_path, device=device)
    
    return weights, config


# =============================================================================
# Option 3: Integration with your existing code structure
# =============================================================================

class GemmaScope2CLTManager:
    """
    Manager class for Gemma Scope 2 CLTs, similar to your ReplacementModelManager.
    """
    
    # Available CLT repos (CLTs primarily for 270m and 1b)
    CLT_REPOS = {
        "270m-pt": "google/gemma-scope-2-270m-pt",
        "270m-it": "google/gemma-scope-2-270m-it",
        "1b-pt": "google/gemma-scope-2-1b-pt",
        "1b-it": "google/gemma-scope-2-1b-it",
        # Larger models also have CLT folders but may have different structure
        "4b-it": "google/gemma-scope-2-4b-it",
        "12b-it": "google/gemma-scope-2-12b-it",
        "27b-it": "google/gemma-scope-2-27b-it",
    }
    
    # Model configs
    MODEL_CONFIGS = {
        "270m": {"n_layers": 18, "d_model": 1536},  # Approximate
        "1b": {"n_layers": 26, "d_model": 2048},    # Approximate
        "4b": {"n_layers": 34, "d_model": 3072},    # Approximate
        "12b": {"n_layers": 48, "d_model": 4096},   # Approximate
        "27b": {"n_layers": 62, "d_model": 5120},   # Approximate
    }
    
    def __init__(
        self,
        model_size: str = "1b-it",
        clt_config: str = "width_262k_l0_medium",
        device: str = None,
    ):
        """
        Initialize the CLT manager.
        
        Args:
            model_size: One of the keys in CLT_REPOS
            clt_config: CLT configuration (e.g., "width_262k_l0_medium")
            device: Device to use
        """
        self.model_size = model_size
        self.clt_config = clt_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_id = self.CLT_REPOS[model_size]
        
        self._clt = None
        self._weights = None
        self._config = None
    
    def load_clt(self) -> tuple[dict, dict]:
        """Load CLT weights and config."""
        if self._weights is None:
            print(f"Loading CLT from {self.repo_id}/clt/{self.clt_config}")
            self._weights, self._config = load_clt_from_huggingface(
                repo_id=self.repo_id,
                clt_path=f"clt/{self.clt_config}",
                device=self.device,
            )
        return self._weights, self._config
    
    def get_encoder_weights(self) -> torch.Tensor:
        """Get encoder weights from loaded CLT."""
        weights, _ = self.load_clt()
        # Weight names may vary - common patterns:
        possible_names = ["W_enc", "encoder.weight", "w_enc"]
        for name in possible_names:
            if name in weights:
                return weights[name]
        print(f"Available weight keys: {list(weights.keys())}")
        raise KeyError("Could not find encoder weights")
    
    def encode_activations(
        self,
        activations: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """
        Encode activations using the CLT encoder.
        
        Args:
            activations: Input activations (batch, seq_len, d_model)
            layer: Layer index
            
        Returns:
            Feature activations
        """
        weights, config = self.load_clt()
        
        # Get encoder weights and bias
        W_enc = weights.get("W_enc", weights.get("encoder.weight"))
        b_enc = weights.get("b_enc", weights.get("encoder.bias"))
        threshold = config.get("threshold", 0.0)
        
        # Handle different weight shapes
        if W_enc.dim() == 2:
            # Single encoder for all layers
            pre_acts = torch.einsum("bsd,fd->bsf", activations, W_enc)
        else:
            # Per-layer encoder
            pre_acts = torch.einsum("bsd,fd->bsf", activations, W_enc[layer])
        
        if b_enc is not None:
            if b_enc.dim() == 1:
                pre_acts = pre_acts + b_enc
            else:
                pre_acts = pre_acts + b_enc[layer]
        
        # Apply JumpReLU or ReLU
        if threshold > 0:
            features = torch.where(
                pre_acts > threshold,
                pre_acts,
                torch.zeros_like(pre_acts)
            )
        else:
            features = torch.relu(pre_acts)
        
        return features
    
    def compute_l0_per_layer(
        self,
        features: torch.Tensor,
        skip_bos: bool = True,
    ) -> torch.Tensor:
        """
        Compute L0 (number of active features) per layer.
        
        Args:
            features: Feature activations (batch, n_layers, seq_len, n_features)
                      or (n_layers, seq_len, n_features)
            skip_bos: Whether to skip the BOS token
            
        Returns:
            L0 per layer
        """
        is_single = features.dim() == 3
        if is_single:
            features = features.unsqueeze(0)
        
        non_zero = (features > 0).float()
        l0_per_token = non_zero.sum(dim=-1)
        
        if skip_bos:
            l0_per_layer = l0_per_token[:, :, 1:].mean(dim=-1)
        else:
            l0_per_layer = l0_per_token.mean(dim=-1)
        
        if is_single:
            return l0_per_layer.squeeze(0)
        return l0_per_layer


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example of how to use the CLT manager."""
    
    # Initialize manager
    manager = GemmaScope2CLTManager(
        model_size="1b-it",
        clt_config="width_262k_l0_medium",  # or "width_16k_l0_small", etc.
    )
    
    # Load weights to inspect structure
    weights, config = manager.load_clt()
    print("Config:", config)
    print("Weight keys:", list(weights.keys()))
    for key, val in weights.items():
        print(f"  {key}: {val.shape}")
    
    # To compute L0, you would:
    # 1. Run the base Gemma model and collect residual stream activations
    # 2. Pass them through the CLT encoder
    # 3. Count non-zero features
    
    # Pseudo-code (requires actual model):
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    
    # Tokenize
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, d_model)
    
    # Encode through CLT and compute L0
    all_features = []
    for layer, hs in enumerate(hidden_states[:-1]):  # Skip final layer
        features = manager.encode_activations(hs, layer)
        all_features.append(features)
    
    features_tensor = torch.stack(all_features, dim=1)  # (batch, n_layers, seq, n_features)
    l0_per_layer = manager.compute_l0_per_layer(features_tensor)
    print("L0 per layer:", l0_per_layer)
    """


if __name__ == "__main__":
    # List available CLT configurations
    print("Available Gemma Scope 2 CLT repos:")
    for key, repo in GemmaScope2CLTManager.CLT_REPOS.items():
        print(f"  {key}: {repo}")
    
    print("\n" + "="*60)
    print("To use these CLTs, run: example_usage()")
    print("="*60)
    
    # Uncomment to run example
    # example_usage()
