"""
Runner for L0 analysis using Gemma Scope 2 CLTs.

Uses the model manager from tmp/ with the analysis infrastructure from src/.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.analysis.analysis_config import AnalysisConfig
from src.analysis.config_analysis.config_l0_replacement_model_step import ConfigL0ReplacementModelStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyzer import CrossConfigAnalyzer
from src.analysis_runner import _load_prompts_from_dataset, get_results_base_dir
from src.constants import CONFIG_BASE_DIR, DEFAULT_BATCH_SIZE

from gemma_scope_2_clt_manager import GemmaScope2CLTModelManager


def run_l0_for_config(
        config_dir: Path,
        config_name: str,
        model_variant: str = "1b-it",
        clt_config: str = "width_262k_l0_medium",
        batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Run L0 analysis for a config using Gemma Scope 2 CLTs.

    Args:
        config_dir: Directory containing config files.
        config_name: Name of the config file (without .json extension).
        model_variant: Gemma model variant (e.g., "1b-it", "270m-pt").
        clt_config: CLT configuration name.
        batch_size: Batch size for processing.

    Returns:
        Dictionary mapping SupportedConfigAnalyzeStep to results.
    """
    config_path = config_dir / f"{config_name}.json"
    config = AnalysisConfig.from_file(config_path)

    if config.dataset:
        prompts = _load_prompts_from_dataset(config.dataset)
        print(f"Dataset: {config.dataset.name} ({len(prompts)} samples)")
    else:
        prompts = config.id_to_prompt

    print(f"\nStarting L0 analysis for {config_name}")
    print(f"Model: {model_variant}, CLT: {clt_config}")
    print("==================================")

    manager = GemmaScope2CLTModelManager(
        model_variant=model_variant,
        clt_config=clt_config,
    )

    prompt_ids = list(prompts.keys())
    prompt_texts = list(prompts.values())
    num_prompts = len(prompt_texts)

    # Process in batches to save memory
    all_l0 = []
    for batch_start in range(0, num_prompts, batch_size):
        batch_end = min(batch_start + batch_size, num_prompts)
        batch_prompts = prompt_texts[batch_start:batch_end]

        hidden_states_batch = manager.get_hidden_states_batch(batch_prompts, batch_size=len(batch_prompts))

        # Process each prompt's hidden states
        batch_l0 = []
        for i in range(hidden_states_batch.shape[0]):
            prompt_hs = [hidden_states_batch[i, layer] for layer in range(hidden_states_batch.shape[1])]
            features = manager.encode_features(prompt_hs)
            l0 = ConfigL0ReplacementModelStep.compute_l0_per_layer(features)
            batch_l0.append(l0)

        all_l0.extend(batch_l0)

        del hidden_states_batch, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {pid: all_l0[i] for i, pid in enumerate(prompt_ids)}

    print("==================================")
    print(f"Finished L0 analysis for {len(prompts)} prompts\n")

    return {
        SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: {
            "results": results,
            "d_transcoder": manager.get_d_transcoder(),
        }
    }


def run_l0_for_all_configs(
        config_names: List[str] = None,
        config_dir: str = None,
        model_variant: str = "1b-it",
        clt_config: str = "width_262k_l0_medium",
        batch_size: int = DEFAULT_BATCH_SIZE,
        save_path: Optional[Path] = None,
) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Run L0 analysis across all configs, then run cross-config analysis.

    Args:
        config_names: List of config names to run, or None/'all' for all configs.
        config_dir: Directory containing config files.
        model_variant: Gemma model variant.
        clt_config: CLT configuration name.
        batch_size: Batch size for processing.
        save_path: Optional path for saving results.

    Returns:
        Dictionary of cross-config results from CrossConfigAnalyzer.
    """
    config_dir = Path(config_dir)
    full_config_dir = CONFIG_BASE_DIR / config_dir

    if not config_names or config_names[0].lower().strip() == 'all':
        config_names = [f.stem for f in full_config_dir.glob("*.json")]

    all_config_results = {}

    for config_name in config_names:
        config_name = config_name.strip()
        config_results = run_l0_for_config(
            full_config_dir,
            config_name,
            model_variant=model_variant,
            clt_config=clt_config,
            batch_size=batch_size,
        )
        all_config_results[config_name] = config_results

    print(f"Finished running all {len(config_names)} configs.")

    if save_path is None:
        save_path, _ = get_results_base_dir()

    analyzer = CrossConfigAnalyzer(all_config_results, save_path=save_path)
    return analyzer.run()
