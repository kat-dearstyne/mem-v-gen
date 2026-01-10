import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from src.analysis.analysis_config import AnalysisConfig, Task
from src.analysis.config_analysis.config_analyzer import ConfigAnalyzer
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyzer import CrossConditionAnalyzer
from src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step import INTERSECTION_METRICS_KEY
from src.analysis.cross_config_analysis.cross_config_analyzer import CrossConfigAnalyzer
from src.analysis.cross_config_analysis.cross_config_feature_overlap_step import CrossConfigFeatureOverlapStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CrossConfigSubgraphFilterStep, SHARED_FEATURES_KEY
)
from src.constants import DATA_PATH, MODEL, SUBMODELS, TOP_K, CONFIG_BASE_DIR, OUTPUT_DIR
from src.metrics import FeatureSharingMetrics
from src.neuronpedia_manager import GraphConfig, NeuronpediaManager

# Filenames for cross-config results (re-exported from steps for backward compatibility)
OVERLAP_ANALYSIS_FILENAME = CrossConfigSubgraphFilterStep.OVERLAP_ANALYSIS_FILENAME
FEATURE_OVERLAP_METRICS_FILENAME = CrossConfigFeatureOverlapStep.FEATURE_OVERLAP_METRICS_FILENAME
SHARED_FEATURE_METRICS_FILENAME = CrossConfigSubgraphFilterStep.SHARED_FEATURE_METRICS_FILENAME


def analyze_conditions(condition_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]],
                       save_dir: Optional[Path] = None,
                       condition_order: Optional[List[str]] = None,
                       config_order: Optional[List[str]] = None,
                       extra_series: Optional[dict] = None) -> Dict[str, Any]:
    """
    Analyzes and visualizes results across multiple conditions using CrossConditionAnalyzer.

    Args:
        condition_results: Dictionary mapping condition names to CrossConfigAnalyzer results.
        save_dir: Directory to save visualizations (optional).
        condition_order: Order for conditions in plots (optional).
        config_order: Order for configs in plots (optional).
        extra_series: Optional dict with 'label' and 'data' for extra line in plots.

    Returns:
        Dictionary mapping step names to their results.
    """
    analyzer = CrossConditionAnalyzer(condition_results, save_path=save_dir)
    return analyzer.run(
        condition_order=condition_order,
        config_order=config_order,
        extra_series=extra_series
    )


def load_condition_results_from_dirs(dirs: List[Path]) -> Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]:
    """
    Loads cross-config results from directories and reconstructs condition_results format.

    Each directory is treated as a separate condition. The directory name (first part
    before '-') is used as the condition name.

    Args:
        dirs: List of directories containing cross-config analysis CSV results.

    Returns:
        Dictionary mapping condition names to reconstructed CrossConfigAnalyzer results.
    """
    condition_results = {}

    for dir_path in dirs:
        condition_name = dir_path.name.split("-")[0]
        step_results = {}

        # Load intersection metrics (overlap-analysis.csv)
        overlap_path = dir_path / OVERLAP_ANALYSIS_FILENAME
        if overlap_path.exists():
            intersection_df = pd.read_csv(overlap_path, index_col=0)
            if SupportedConfigAnalyzeStep.SUBGRAPH_FILTER not in step_results:
                step_results[SupportedConfigAnalyzeStep.SUBGRAPH_FILTER] = {}
            step_results[SupportedConfigAnalyzeStep.SUBGRAPH_FILTER][INTERSECTION_METRICS_KEY] = intersection_df

        # Load shared feature metrics
        shared_path = dir_path / SHARED_FEATURE_METRICS_FILENAME
        if shared_path.exists():
            shared_df = pd.read_csv(shared_path, index_col=0)
            if SupportedConfigAnalyzeStep.SUBGRAPH_FILTER not in step_results:
                step_results[SupportedConfigAnalyzeStep.SUBGRAPH_FILTER] = {}
            step_results[SupportedConfigAnalyzeStep.SUBGRAPH_FILTER][SHARED_FEATURES_KEY] = shared_df

        # Load feature overlap metrics
        feature_overlap_path = dir_path / FEATURE_OVERLAP_METRICS_FILENAME
        if feature_overlap_path.exists():
            feature_overlap_df = pd.read_csv(feature_overlap_path, index_col=0)
            step_results[SupportedConfigAnalyzeStep.FEATURE_OVERLAP] = feature_overlap_df

        if step_results:
            condition_results[condition_name] = step_results

    return condition_results


def analyze_conditions_post_run(dirs: List[Path],
                                save_dir: Optional[Path] = None,
                                condition_order: Optional[List[str]] = None,
                                config_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Loads CSVs from multiple directories and generates cross-condition visualizations.

    This is a convenience function that loads saved CSV results and delegates to
    CrossConditionAnalyzer. For programmatic use with in-memory results, use
    analyze_conditions() directly.

    Args:
        dirs: List of directories containing analysis results to compare.
        save_dir: Directory to save visualizations (optional).
        condition_order: Order for conditions in plots (optional).
        config_order: Order for configs in plots (optional).

    Returns:
        Dictionary mapping step names to their results.
    """
    # Load extra series for jaccard line plot (average precision from error analysis)
    extra_series = None
    if dirs:
        error_results_path = dirs[0] / "memorized vs. random" / "error-results-individual.csv"
        if error_results_path.exists():
            error_df = pd.read_csv(error_results_path)
            ap_df = error_df[error_df['metric'] == 'top_k_10']
            extra_series = {
                'label': 'Average Precision',
                'data': ap_df.set_index('sample')['g1_score']
            }

    # Load condition results from directories
    condition_results = load_condition_results_from_dirs(dirs)

    if not condition_results:
        print("No results found in provided directories")
        return {}

    return analyze_conditions(
        condition_results,
        save_dir=save_dir,
        condition_order=condition_order,
        config_order=config_order,
        extra_series=extra_series
    )


def run_for_config(config_dir: Path, config_name: str,
                   run_error_analysis: bool, submodel_num: int = 0
                   ) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Runs the main logic for a specified prompt config.

    Args:
        config_dir: Directory containing config files.
        config_name: Name of the config file (without .json extension).
        run_error_analysis: Whether to run error ranking analysis.
        submodel_num: Index of submodel to use.

    Returns:
        Dictionary mapping SupportedConfigAnalyzeStep to results.
    """
    config_path = config_dir / f"{config_name}.json"
    try:
        config = AnalysisConfig.from_file(config_path)
    except Exception as e:
        print(f"Unable to load {config_name}")
        raise e

    model = MODEL
    submodel = SUBMODELS[submodel_num]
    base_save_path = os.path.expanduser(DATA_PATH)
    graph_dir = os.path.join(base_save_path, "graphs")
    results = {}

    graph_config = GraphConfig(model=model, submodel=submodel)
    neuronpedia_manager = NeuronpediaManager(graph_dir=graph_dir, config=graph_config)
    analyzer = ConfigAnalyzer(neuronpedia_manager, prompts=config.id_to_prompt)

    if config.task == Task.PROMPT_SUBGRAPH_COMPARE:
        print(f"\nStarting run for {config_name} with model {model} and submodel {submodel}")
        print("\n".join([f"{pid}: {p}" if pid != p else p for pid, p in config.id_to_prompt.items()]))
        print("==================================")

        diff_prompt_ids = config.diff_prompt_ids or None
        sim_prompt_ids = config.sim_prompt_ids or None
        comparison_ids = [pid for pid in config.id_to_prompt.keys()
                         if pid != config.main_prompt_id and "rephrased" not in pid]

        steps_to_run = [SupportedConfigAnalyzeStep.SUBGRAPH_FILTER]
        if run_error_analysis:
            steps_to_run.append(SupportedConfigAnalyzeStep.ERROR_RANKING)

        results |= analyzer.run(
            steps_to_run,
            main_prompt_id=config.main_prompt_id,
            prompts_with_unique_features=diff_prompt_ids,
            prompts_with_shared_features=sim_prompt_ids,
            comparison_prompt_ids=comparison_ids,
            use_same_token=True
        )
        print("==================================\n")
    elif config.task == Task.TOKEN_SUBGRAPH_COMPARE:
        print(f"\nStarting run with model {model} and submodel {submodel}"
              f"\nPrompt1: '{config.main_prompt}'\n"
              f"\nToken of interest: {config.token_of_interest}.\n")

        results |= analyzer.run(
            SupportedConfigAnalyzeStep.TOKEN_SUBGRAPH,
            prompt_id=config.main_prompt_id,
            token_of_interest=config.token_of_interest,
            top_k_tokens=TOP_K
        )
    elif config.task == Task.COMBINED_COMPARE:
        print(f"\nStarting combined comparison for {config_name} with model {model} and submodel {submodel}")
        print("==================================")

        if not config.diff_prompts:
            raise ValueError("COMBINED_COMPARE task requires DIFF_PROMPTS to be specified")

        results |= analyzer.run(
            SupportedConfigAnalyzeStep.FEATURE_OVERLAP,
            main_prompt_id=config.main_prompt_id,
            comparison_prompt_ids=config.diff_prompt_ids,
            debug=False,
            filter_by_act_density=50
        )
        combined_metrics = results[SupportedConfigAnalyzeStep.FEATURE_OVERLAP]
        print(f"Combined metrics: unique_frac={combined_metrics[FeatureSharingMetrics.UNIQUE_FRAC]:.3f}, "
              f"shared_frac={combined_metrics[FeatureSharingMetrics.SHARED_FRAC]:.3f}")
        print("==================================\n")
    elif config.task == Task.REPLACEMENT_MODEL:
        print(f"\nStarting replacement model analysis for {config_name} with model {model} and submodel {submodel}")
        print("==================================")

        results |= analyzer.run(
            SupportedConfigAnalyzeStep.REPLACEMENT_MODEL,
            memorized_completion=config.memorized_completion
        )
        print("==================================\n")
    elif config.task == Task.EARLY_LAYER_CONTRIBUTION:
        print(f"\nStarting early layer contribution analysis for {config_name} with model {model} and submodel {submodel}")
        print("==================================")

        results |= analyzer.run(
            SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION
        )
        print("==================================\n")
    else:
        raise NotImplementedError(f"Unknown task: {config.task}")

    return results


def get_results_base_dir() -> Path:
    """
    Get the base directory for saving results.

    Uses RESULTS_DIR env var if set, otherwise generates from current datetime.

    Returns:
        Path to the results base directory.
    """
    results_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dirname := os.getenv("RESULTS_DIR", "").strip():
        results_dir = dirname
    return OUTPUT_DIR / results_dir


def run_for_all_configs(config_names: List[str] = None, config_dir: str = None,
                        run_error_analysis: bool = False, submodel_num: int = 0,
                        save_path: Optional[Path] = None
                        ) -> Dict[SupportedConfigAnalyzeStep, Any]:
    """
    Runs analysis across all configs. The task type (prompt, token, or combined)
    is determined by each config's TASK field.

    Args:
        config_names: List of config names to run, or None/'all' for all configs.
        config_dir: Directory containing config files.
        run_error_analysis: Whether to run error ranking analysis.
        submodel_num: Index of submodel to use.
        save_path: Optional path for saving results. If not provided, generates from
            RESULTS_DIR env var or current datetime.

    Returns:
        Dictionary of cross-config results from CrossConfigAnalyzer.
    """
    assert config_names or config_dir, "Must provide config names or config dir!"
    config_dir = Path(config_dir)
    if config_dir:
        config_dir = CONFIG_BASE_DIR / config_dir
    if not config_names or config_names[0].lower().strip() == 'all':
        config_names = [f.stem for f in config_dir.glob("*.json")]

    all_config_results = {}

    for config in config_names:
        config = config.strip()
        config_results = run_for_config(
            config_dir, config, submodel_num=submodel_num,
            run_error_analysis=run_error_analysis
        )
        all_config_results[config] = config_results

    if save_path is None:
        save_path = get_results_base_dir()

    analyzer = CrossConfigAnalyzer(all_config_results, save_path=save_path)
    return analyzer.run()


def run_cross_condition_analysis(
        config_dirs: Optional[List[str]] = None,
        config_names: Optional[List[str]] = None,
        submodel_nums: Optional[List[int]] = None,
        run_error_analysis: bool = False,
        condition_order: Optional[List[str]] = None,
        config_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Runs cross-condition analysis by iterating through multiple config directories
    or submodel parameters and comparing results across conditions.

    The conditions are determined by which parameter has multiple values:
    - If config_dirs has >1 entry, iterates over config dirs (uses first submodel_num)
    - If submodel_nums has >1 entry, iterates over submodels (uses first config_dir)

    Results are saved in a base directory (from RESULTS_DIR env var or datetime)
    with subdirectories for each condition.

    Args:
        config_dirs: List of config directories. If >1, each becomes a condition.
        config_names: List of config names to run, or None/'all' for all configs.
        submodel_nums: List of submodel indices. If >1, each becomes a condition.
        run_error_analysis: Whether to run error ranking analysis.
        condition_order: Order for conditions in plots (optional).
        config_order: Order for configs in plots (optional).

    Returns:
        Dictionary mapping step names to their results.
    """
    condition_results = {}
    config_dirs = config_dirs or []
    submodel_nums = submodel_nums or [0]

    # Create base results directory for all conditions
    base_results_dir = get_results_base_dir()

    if len(config_dirs) > 1:
        # Iterate over config directories, use single submodel
        submodel_num = submodel_nums[0]
        for config_dir in config_dirs:
            condition_name = Path(config_dir).name
            print(f"\n{'='*50}")
            print(f"Running condition: {condition_name}")
            print(f"{'='*50}")

            cross_config_results = run_for_all_configs(
                config_names=config_names,
                config_dir=config_dir,
                run_error_analysis=run_error_analysis,
                submodel_num=submodel_num,
                save_path=base_results_dir / condition_name
            )
            condition_results[condition_name] = cross_config_results

    elif len(submodel_nums) > 1:
        # Iterate over submodels, use single config_dir
        config_dir = config_dirs[0] if config_dirs else ""

        for submodel_num in submodel_nums:
            condition_name = SUBMODELS[submodel_num]
            print(f"\n{'='*50}")
            print(f"Running condition: {condition_name}")
            print(f"{'='*50}")

            cross_config_results = run_for_all_configs(
                config_names=config_names,
                config_dir=config_dir,
                run_error_analysis=run_error_analysis,
                submodel_num=submodel_num,
                save_path=base_results_dir / condition_name
            )
            condition_results[condition_name] = cross_config_results

    else:
        raise ValueError("Must provide multiple config_dirs or multiple submodel_nums")

    if not condition_results:
        print("No condition results to analyze")
        return {}

    # Save cross-condition analysis in base directory
    return analyze_conditions(
        condition_results,
        save_dir=base_results_dir,
        condition_order=condition_order,
        config_order=config_order
    )
