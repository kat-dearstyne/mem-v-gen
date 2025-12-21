from datetime import datetime
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from common_utils import user_select_models, user_select_prompt
from constants import PROMPT_IDS_MEMORIZED, PROMPT_ID_BASELINE, DATA_PATH, MODEL, SUBMODELS, TOP_K, CONFIG_BASE_DIR, \
    OUTPUT_DIR, OVERLAP_ANALYSIS_FILENAME, FEATURE_LAYER, FEATURE_ID
from error_comparisons import run_error_ranking, analyze_conditions
from subgraph_comparisons import (compare_prompt_subgraphs, compare_token_subgraphs, compare_prompts_combined,
                                  IntersectionMetrics, CombinedMetrics, SharedFeatureMetrics)
from visualizations import (
    visualize_feature_presence,
    plot_metric_by_condition,
    plot_jaccard_by_condition,
    plot_metric_heatmap,
    plot_jaccard_heatmap,
    plot_metric_boxplot,
    plot_jaccard_boxplot,
    plot_metric_line,
    plot_jaccard_line,
    plot_metric_vs_probability_scatter,
    plot_metric_vs_probability_combined,
    plot_probability_by_condition,
    plot_correlation_heatmap,
    plot_combined_metrics,
    plot_shared_feature_metrics,
)


class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"
    COMBINED_COMPARE = "combined"


COMBINED_OVERLAP_METRICS_FILENAME = "combined-overlap-metrics.csv"
SHARED_FEATURE_METRICS_FILENAME = "shared-feature-metrics.csv"


def load_and_combine_csvs(dirs: List[Path], filename: str,
                          normalize_config_names: bool = True,
                          add_prompt_type: bool = False) -> pd.DataFrame:
    """
    Loads CSV files from multiple directories and combines them into a single DataFrame.

    Args:
        dirs: List of directories to load CSVs from
        filename: Name of the CSV file to load from each directory
        normalize_config_names: If True, strip directory prefixes from config names
        add_prompt_type: If True, add a prompt_type column based on directory name
    """
    dfs = []
    for dir_path in dirs:
        path = dir_path / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if add_prompt_type:
            # Use first part of directory name (before '-') as prompt type
            df["prompt_type"] = dir_path.name.split("-")[0]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    if normalize_config_names:
        # Strip directory prefixes (e.g., "baseline/biblical" -> "biblical")
        combined['config_name'] = combined['config_name'].apply(
            lambda x: x.split('/')[-1] if '/' in str(x) else x
        )

    return combined


def analyze_overlap(dirs: List[Path],
                    save_dir: Optional[Path] = None,
                    condition_order: Optional[List[str]] = None,
                    config_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Main function to load CSVs from multiple directories, combine them, and generate all visualizations.

    Args:
        dirs: List of directories containing analysis results to compare
        save_dir: Directory to save visualizations (optional)
        condition_order: Order for conditions in plots (optional)
        config_order: Order for configs in plots (optional)
    """
    df = load_and_combine_csvs(dirs, filename=OVERLAP_ANALYSIS_FILENAME)
    has_overlap_metrics = not df.empty

    # Load combined metrics if available
    combined_df = load_and_combine_csvs(dirs, filename=COMBINED_OVERLAP_METRICS_FILENAME, add_prompt_type=True)
    if combined_df.empty:
        combined_df = None

    # Load shared feature metrics if available
    shared_feature_df = load_and_combine_csvs(dirs, filename=SHARED_FEATURE_METRICS_FILENAME, add_prompt_type=True)
    if shared_feature_df.empty:
        shared_feature_df = None

    # Derive condition_order and config_order from available data
    if condition_order is None:
        if has_overlap_metrics:
            condition_order = sorted(df['prompt_type'].unique().tolist())
        elif combined_df is not None:
            condition_order = sorted(combined_df['prompt_type'].unique().tolist())
        elif shared_feature_df is not None:
            condition_order = sorted(shared_feature_df['prompt_type'].unique().tolist())

    if config_order is None:
        if has_overlap_metrics:
            config_order = sorted(df['config_name'].unique().tolist())
        elif combined_df is not None:
            config_order = sorted(combined_df['config_name'].unique().tolist())
        elif shared_feature_df is not None:
            config_order = sorted(shared_feature_df['config_name'].unique().tolist())

    # Check for error-results-individual.csv in first dir to add average_precision line
    ap_series = None
    error_results_path = dirs[0] / "memorized vs. random" / "error-results-individual.csv"
    if error_results_path.exists():
        error_df = pd.read_csv(error_results_path)
        ap_df = error_df[error_df['metric'] == 'top_k_10']
        ap_series = {
            'label': 'Average Precision',
            'data': ap_df.set_index('sample')['g1_score']
        }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if has_overlap_metrics:
            # Jaccard index visualizations
            plot_jaccard_by_condition(df, save_path=save_dir / "jaccard_bar.png",
                                      condition_order=condition_order, config_order=config_order)
            plot_jaccard_heatmap(df, save_path=save_dir / "jaccard_heatmap.png",
                                 condition_order=condition_order, config_order=config_order)
            plot_jaccard_boxplot(df, save_path=save_dir / "jaccard_boxplot.png",
                                 condition_order=condition_order)
            plot_jaccard_line(df, save_path=save_dir / "jaccard_line.png",
                              condition_order=condition_order, config_order=config_order,
                              extra_series=ap_series)

            # Weighted Jaccard visualizations
            plot_metric_by_condition(df, metric_col='relative_jaccard',
                                     title='Weighted Jaccard by Condition',
                                     ylabel='Weighted Jaccard',
                                     save_path=save_dir / "relative_jaccard_bar.png",
                                     condition_order=condition_order, config_order=config_order)
            plot_metric_heatmap(df, metric_col='relative_jaccard',
                                title='Weighted Jaccard Heatmap',
                                cbar_label='Weighted Jaccard',
                                save_path=save_dir / "relative_jaccard_heatmap.png",
                                condition_order=condition_order, config_order=config_order)
            plot_metric_boxplot(df, metric_col='relative_jaccard',
                                title='Weighted Jaccard Distribution by Condition',
                                ylabel='Weighted Jaccard',
                                save_path=save_dir / "relative_jaccard_boxplot.png",
                                condition_order=condition_order)
            plot_metric_line(df, metric_col='relative_jaccard',
                             title='Weighted Jaccard by Config',
                             ylabel='Weighted Jaccard',
                             save_path=save_dir / "relative_jaccard_line.png",
                             condition_order=condition_order, config_order=config_order)

            # Output probability visualizations
            plot_probability_by_condition(df, save_path=save_dir / "output_probability_boxplot.png",
                                          condition_order=condition_order)

            # Jaccard vs Output Probability relationship visualizations
            plot_metric_vs_probability_scatter(df, metric_col='jaccard_index',
                                               title='Jaccard Index vs Output Probability',
                                               xlabel='Jaccard Index',
                                               save_path=save_dir / "jaccard_vs_probability.png",
                                               condition_order=condition_order)
            plot_metric_vs_probability_scatter(df, metric_col='relative_jaccard',
                                               title='Weighted Jaccard vs Output Probability',
                                               xlabel='Weighted Jaccard',
                                               save_path=save_dir / "relative_jaccard_vs_probability.png",
                                               condition_order=condition_order)
            plot_metric_vs_probability_combined(df, save_path=save_dir / "jaccard_metrics_vs_probability.png",
                                                condition_order=condition_order)
            plot_correlation_heatmap(df, save_path=save_dir / "metric_correlations.png",
                                     condition_order=condition_order)

            # Also save the combined CSV
            df.to_csv(save_dir / "combined_results.csv", index=False)

        # Plot combined metrics if available
        if combined_df is not None:
            plot_combined_metrics(combined_df, save_dir=save_dir, config_order=config_order)

        # Plot shared feature metrics if available
        if shared_feature_df is not None:
            plot_shared_feature_metrics(shared_feature_df, save_dir=save_dir, config_order=config_order)

        print(f"Saved overlap analysis results to: {save_dir}")
    else:
        if has_overlap_metrics:
            # Jaccard index visualizations
            plot_jaccard_by_condition(df, condition_order=condition_order, config_order=config_order)
            plot_jaccard_heatmap(df, condition_order=condition_order, config_order=config_order)
            plot_jaccard_boxplot(df, condition_order=condition_order)
            plot_jaccard_line(df, condition_order=condition_order, config_order=config_order,
                              extra_series=ap_series)

            # Weighted Jaccard visualizations
            plot_metric_by_condition(df, metric_col='relative_jaccard',
                                     title='Weighted Jaccard by Condition',
                                     ylabel='Weighted Jaccard',
                                     condition_order=condition_order, config_order=config_order)
            plot_metric_heatmap(df, metric_col='relative_jaccard',
                                title='Weighted Jaccard Heatmap',
                                cbar_label='Weighted Jaccard',
                                condition_order=condition_order, config_order=config_order)
            plot_metric_boxplot(df, metric_col='relative_jaccard',
                                title='Weighted Jaccard Distribution by Condition',
                                ylabel='Weighted Jaccard',
                                condition_order=condition_order)
            plot_metric_line(df, metric_col='relative_jaccard',
                             title='Weighted Jaccard by Config',
                             ylabel='Weighted Jaccard',
                             condition_order=condition_order, config_order=config_order)

            # Output probability visualizations
            plot_probability_by_condition(df, condition_order=condition_order)

            # Jaccard vs Output Probability relationship visualizations
            plot_metric_vs_probability_scatter(df, metric_col='jaccard_index',
                                               title='Jaccard Index vs Output Probability',
                                               xlabel='Jaccard Index',
                                               condition_order=condition_order)
            plot_metric_vs_probability_scatter(df, metric_col='relative_jaccard',
                                               title='Weighted Jaccard vs Output Probability',
                                               xlabel='Weighted Jaccard',
                                               condition_order=condition_order)
            plot_metric_vs_probability_combined(df, condition_order=condition_order)
            plot_correlation_heatmap(df, condition_order=condition_order)

        # Plot combined metrics if available
        if combined_df is not None:
            plot_combined_metrics(combined_df, config_order=config_order)

        # Plot shared feature metrics if available
        if shared_feature_df is not None:
            plot_shared_feature_metrics(shared_feature_df, config_order=config_order)

    return df


def run_for_config(config_dir: Path, config_name: str,
                   run_error_analysis: bool, submodel_num: int = 0
                   ) -> tuple[Optional[dict], Optional[dict[str, dict]], Optional[CombinedMetrics], Optional[dict], Optional[SharedFeatureMetrics]]:
    """
    Runs the main logic for a specified prompt config.
    Returns (pairwise_metrics, error_analysis_results, combined_metrics, feature_presence, shared_feature_metrics).
    """
    config_path = config_dir / f"{config_name}.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Unable to load {config_name}")
        raise e

    # Load prompts from config
    main_prompt = config["MAIN_PROMPT"]
    diff_prompts = config.get("DIFF_PROMPTS", [])
    sim_prompts = config.get("SIM_PROMPTS", [])
    token_of_interest = config.get("TOKEN_OF_INTEREST")
    selected_task = config.get("TASK")

    if diff_prompts or sim_prompts:
        assert not token_of_interest or selected_task, ("Both TOKEN_OF_INTEREST and DIFF_PROMPTS/SIM_PROMPTS supplied. "
                                                        "Must specify what task to perform.")
        selected_task = selected_task if selected_task else Task.PROMPT_SUBGRAPH_COMPARE

    elif token_of_interest:
        selected_task = selected_task if selected_task else Task.TOKEN_SUBGRAPH_COMPARE

    base_save_path = os.path.expanduser(DATA_PATH)
    model, submodel = user_select_models(model=MODEL, submodel=SUBMODELS[submodel_num])
    graph_dir = os.path.join(base_save_path, "graphs")

    prompt = user_select_prompt(prompt_default=main_prompt, graph_dir=graph_dir)
    all_prompts = [prompt] + diff_prompts + sim_prompts
    prompt_ids = PROMPT_IDS_MEMORIZED if len(config_dir.parents) == 1 else PROMPT_ID_BASELINE
    prompt2ids = {p: (prompt_ids[index] if len(prompt_ids) > index else p)
                  for index, p in enumerate(all_prompts)}
    error_analysis_results = None
    combined_metrics = None
    metrics = None
    feature_presence = None
    shared_feature_metrics = None

    if selected_task == Task.PROMPT_SUBGRAPH_COMPARE:
        print(f"\nStarting run for {config_name} with model {model} and submodel {submodel}")
        print("\n".join([f"{p_id}: {p}" if p_id != p else p for p, p_id in prompt2ids.items()]))
        print("==================================")

        metrics, feature_presence, shared_feature_metrics = compare_prompt_subgraphs(main_prompt=prompt, diff_prompts=diff_prompts, sim_prompts=sim_prompts,
                                           model=model, submodel=submodel, graph_dir=graph_dir, filter_by_act_density=None,
                                           debug=False)
        if metrics:
            metrics = {prompt2ids.get(p, p): res for p, res in metrics.items()}
        if feature_presence:
            feature_presence = {prompt2ids.get(p, p): res for p, res in feature_presence.items()}
        if run_error_analysis:
            error_analysis_results = {}
            for p, p_id in prompt2ids.items():
                if p == prompt or "rephrased" in p_id:
                    continue
                err_res = run_error_ranking(prompt1=prompt, prompt2=p, model=model,
                                            submodel=submodel, graph_dir=graph_dir,
                                            use_same_token=False)
                error_analysis_results[p_id] = err_res
        print("==================================\n")
    elif selected_task == Task.TOKEN_SUBGRAPH_COMPARE:
        print(f"\nStarting run with model {model} and submodel {submodel}"
              f"\nPrompt1: '{prompt}'\n"
              f"\nToken of interest: {token_of_interest}.\n")

        metrics = compare_token_subgraphs(main_prompt=prompt, token_of_interest=token_of_interest, model=model,
                                          submodel=submodel,
                                          graph_dir=graph_dir, tok_k_outputs=TOP_K)
    elif selected_task == Task.COMBINED_COMPARE:
        print(f"\nStarting combined comparison for {config_name} with model {model} and submodel {submodel}")
        print("==================================")

        if not diff_prompts:
            raise ValueError("COMBINED_COMPARE task requires DIFF_PROMPTS to be specified")

        combined_metrics = compare_prompts_combined(
            main_prompt=prompt, diff_prompts=diff_prompts,
            model=model, submodel=submodel, graph_dir=graph_dir, debug=False,
            filter_by_act_density=50
        )
        print(f"Combined metrics: unique_frac={combined_metrics.unique_frac:.3f}, "
              f"shared_frac={combined_metrics.shared_frac:.3f}")
        print("==================================\n")
    else:
        raise NotImplementedError(f"Unknown task: {selected_task}")

    return metrics, error_analysis_results, combined_metrics, feature_presence, shared_feature_metrics


def run_for_all_configs(config_names: List[str] = None, config_dir: str = None,
                        run_error_analysis: bool = False, submodel_num: int = 0):
    """
    Runs analysis across all configs. The task type (prompt, token, or combined)
    is determined by each config's TASK field.
    """
    assert config_names or config_dir, "Must provide config names or config dir!"
    config_dir = Path(config_dir)
    if config_dir:
        config_dir = Path(CONFIG_BASE_DIR) / config_dir
    if not config_names or config_names[0].lower().strip() == 'all':
        config_names = [f.stem for f in config_dir.glob("*.json")]
    all_results = {"config_name": [], "prompt_type": [],
                   **{metric_name: [] for metric_name in IntersectionMetrics._fields}}
    combined_results = {"config_name": [],
                        **{metric_name: [] for metric_name in CombinedMetrics._fields}}
    shared_feature_results = {"config_name": [],
                              **{metric_name: [] for metric_name in SharedFeatureMetrics._fields}}
    feature_results = {"config_name": [], "prompt_id": [], "feature_present": [], "output_prob": []}
    error_pair_results = {}

    for config in config_names:
        config = config.strip()
        results, error_analysis_results, combined_metrics, feature_presence, shared_feature_metrics = run_for_config(
            config_dir, config, submodel_num=submodel_num,
            run_error_analysis=run_error_analysis
        )
        results: dict[str, IntersectionMetrics]
        if error_analysis_results:
            error_pair_results[config] = error_analysis_results
        if results:
            for i, (prompt, metrics) in enumerate(results.items()):
                all_results["config_name"].append(config)
                all_results["prompt_type"].append(prompt)
                for metric_name, metric in zip(IntersectionMetrics._fields, metrics):
                    all_results[metric_name].append(metric)
        if combined_metrics:
            combined_results["config_name"].append(config)
            for metric_name, metric in zip(CombinedMetrics._fields, combined_metrics):
                combined_results[metric_name].append(metric)
        if shared_feature_metrics:
            shared_feature_results["config_name"].append(config)
            for metric_name, metric in zip(SharedFeatureMetrics._fields, shared_feature_metrics):
                shared_feature_results[metric_name].append(metric)
        if feature_presence:
            for prompt_id, present in feature_presence.items():
                feature_results["config_name"].append(config)
                feature_results["prompt_id"].append(prompt_id)
                feature_results["feature_present"].append(present)
                if results and (metrics := results.get(prompt_id)):
                    feature_results["output_prob"].append(metrics.output_probability)
                else:
                    feature_results["output_prob"].append(None)
    results_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(OUTPUT_DIR) / results_dir
    os.makedirs(save_path, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(save_path / OVERLAP_ANALYSIS_FILENAME)

    if combined_results["config_name"]:
        combined_df = pd.DataFrame(combined_results)
        combined_df.to_csv(save_path / COMBINED_OVERLAP_METRICS_FILENAME)

    if shared_feature_results["config_name"]:
        shared_feature_df = pd.DataFrame(shared_feature_results)
        shared_feature_df.to_csv(save_path / SHARED_FEATURE_METRICS_FILENAME)

    if feature_results["config_name"]:
        feature_df = pd.DataFrame(feature_results)
        feature_df.to_csv(save_path / f"feature_{FEATURE_LAYER}_{FEATURE_ID}.csv", index=False)
        visualize_feature_presence(feature_df, save_path)

    if error_pair_results:
        analyze_conditions(error_pair_results, save_path=save_path)
