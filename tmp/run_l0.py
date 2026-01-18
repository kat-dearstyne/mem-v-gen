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
from src.constants import CONFIG_BASE_DIR

from gemma_scope_2_clt_manager import GemmaScope2CLTModelManager


def run_l0_for_config(
        config_dir: Path,
        config_name: str,
        model_variant: str = "1b-it",
        clt_config: str = "width_262k_l0_medium",
) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Run L0 analysis for a config using Gemma Scope 2 CLTs.

    Args:
        config_dir: Directory containing config files.
        config_name: Name of the config file (without .json extension).
        model_variant: Gemma model variant (e.g., "1b-it", "270m-pt").
        clt_config: CLT configuration name.

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

    # Process one prompt at a time to minimize memory usage
    all_l0 = []
    for i, prompt_text in enumerate(prompt_texts):
        hidden_states = manager.get_hidden_states(prompt_text)
        features = manager.encode_features(hidden_states)
        l0 = ConfigL0ReplacementModelStep.compute_l0_per_layer(features)
        all_l0.append(l0.cpu())

        # Clean up GPU memory after each prompt
        del hidden_states, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_prompts} prompts")

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
        save_path: Optional[Path] = None,
) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Run L0 analysis across all configs, then run cross-config analysis.

    Args:
        config_names: List of config names to run, or None/'all' for all configs.
        config_dir: Directory containing config files.
        model_variant: Gemma model variant.
        clt_config: CLT configuration name.
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
        )
        all_config_results[config_name] = config_results

    print(f"Finished running all {len(config_names)} configs.")

    if save_path is None:
        save_path, _ = get_results_base_dir()

    analyzer = CrossConfigAnalyzer(all_config_results, save_path=save_path)
    return analyzer.run()
