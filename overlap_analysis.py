from datetime import datetime
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common_utils import user_select_models, user_select_prompt
from constants import PROMPT_IDS_MEMORIZED, PROMPT_ID_BASELINE, DATA_PATH, MODEL, SUBMODELS, TOP_K, CONFIG_BASE_DIR, \
    OUTPUT_DIR, OVERLAP_ANALYSIS_FILENAME, COLORS, CUSTOM_PALETTE
from error_comparisons import run_error_ranking, analyze_conditions
from subgraph_comparisons import compare_prompt_subgraphs, compare_token_subgraphs, IntersectionMetrics

# Set up plot styling
sns.set_theme(style="whitegrid")
sns.set_palette(CUSTOM_PALETTE)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
})


class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"


def load_and_combine_csvs(dir1: Path, dir2: Path, filename: str,
                          normalize_config_names: bool = True) -> pd.DataFrame:
    """
    Loads two CSV files and combines them into a single DataFrame.
    """
    path1 = dir1 / filename
    path2 = dir2 / filename

    df1 = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)

    combined = pd.concat([df1, df2], ignore_index=True)

    if normalize_config_names:
        # Strip directory prefixes (e.g., "baseline/biblical" -> "biblical")
        combined['config_name'] = combined['config_name'].apply(
            lambda x: x.split('/')[-1] if '/' in str(x) else x
        )

    return combined


def plot_metric_by_condition(df: pd.DataFrame,
                             metric_col: str,
                             title: str,
                             ylabel: str,
                             save_path: Optional[Path] = None,
                             condition_order: Optional[List[str]] = None,
                             config_order: Optional[List[str]] = None) -> None:
    """
    Creates a grouped bar chart comparing a metric across conditions for each config.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()
    if config_order is None:
        config_order = df['config_name'].unique().tolist()

    # Pivot for easier plotting
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values=metric_col)
    pivot_df = pivot_df.reindex(config_order)
    pivot_df = pivot_df[condition_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(config_order))
    n_conditions = len(condition_order)
    width = 0.8 / n_conditions

    palette = CUSTOM_PALETTE[:n_conditions]

    for i, condition in enumerate(condition_order):
        offset = (i - n_conditions / 2 + 0.5) * width
        ax.bar(x + offset, pivot_df[condition], width,
               label=condition, color=palette[i], edgecolor='white', linewidth=0.8)

    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, None)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_jaccard_by_condition(df: pd.DataFrame,
                              title: str = "Jaccard Index by Condition",
                              save_path: Optional[Path] = None,
                              condition_order: Optional[List[str]] = None,
                              config_order: Optional[List[str]] = None) -> None:
    """
    Creates a grouped bar chart comparing Jaccard index across conditions for each config.
    """
    plot_metric_by_condition(df, metric_col='jaccard_index', title=title,
                             ylabel='Jaccard Index', save_path=save_path,
                             condition_order=condition_order, config_order=config_order)


def plot_metric_heatmap(df: pd.DataFrame,
                        metric_col: str,
                        title: str,
                        cbar_label: str,
                        save_path: Optional[Path] = None,
                        condition_order: Optional[List[str]] = None,
                        config_order: Optional[List[str]] = None) -> None:
    """
    Creates a heatmap of metric values across configs and conditions.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()
    if config_order is None:
        config_order = df['config_name'].unique().tolist()

    pivot_df = df.pivot(index='config_name', columns='prompt_type', values=metric_col)
    pivot_df = pivot_df.reindex(config_order)
    pivot_df = pivot_df[condition_order]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd',
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': cbar_label, 'shrink': 0.8},
                ax=ax)

    ax.set_title(title, pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Config', labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_jaccard_heatmap(df: pd.DataFrame,
                         title: str = "Jaccard Index Heatmap",
                         save_path: Optional[Path] = None,
                         condition_order: Optional[List[str]] = None,
                         config_order: Optional[List[str]] = None) -> None:
    """
    Creates a heatmap of Jaccard index values across configs and conditions.
    """
    plot_metric_heatmap(df, metric_col='jaccard_index', title=title,
                        cbar_label='Jaccard Index', save_path=save_path,
                        condition_order=condition_order, config_order=config_order)


def plot_metric_boxplot(df: pd.DataFrame,
                        metric_col: str,
                        title: str,
                        ylabel: str,
                        save_path: Optional[Path] = None,
                        condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a boxplot comparing metric distributions across conditions.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    palette = CUSTOM_PALETTE[:len(condition_order)]

    sns.boxplot(x='prompt_type', y=metric_col, data=df,
                order=condition_order, palette=palette, width=0.5,
                linewidth=1.5, fliersize=0, ax=ax)
    sns.stripplot(x='prompt_type', y=metric_col, data=df,
                  order=condition_order, color='#333333', alpha=0.6,
                  size=6, jitter=0.15, ax=ax)

    ax.set_title(title, pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_jaccard_boxplot(df: pd.DataFrame,
                         title: str = "Jaccard Index Distribution by Condition",
                         save_path: Optional[Path] = None,
                         condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a boxplot comparing Jaccard index distributions across conditions.
    """
    plot_metric_boxplot(df, metric_col='jaccard_index', title=title,
                        ylabel='Jaccard Index', save_path=save_path,
                        condition_order=condition_order)


def plot_metric_line(df: pd.DataFrame,
                     metric_col: str,
                     title: str,
                     ylabel: str,
                     save_path: Optional[Path] = None,
                     condition_order: Optional[List[str]] = None,
                     config_order: Optional[List[str]] = None,
                     extra_series: Optional[dict] = None) -> None:
    """
    Creates a line plot showing metric trends across configs for each condition.

    Args:
        extra_series: Optional dict with 'label' and 'data' (pd.Series indexed by config_name)
                      to plot an additional line on the chart.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()
    if config_order is None:
        config_order = df['config_name'].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = CUSTOM_PALETTE[:len(condition_order)]

    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name')
        cond_df = cond_df.reindex(config_order)

        ax.plot(config_order, cond_df[metric_col],
                marker='o', markersize=8, linewidth=2,
                color=palette[i], label=condition)

    # Plot extra series if provided
    if extra_series is not None:
        extra_data = extra_series['data'].reindex(config_order)
        ax.plot(config_order, extra_data,
                marker='s', markersize=8, linewidth=2, linestyle='--',
                color='#333333', label=extra_series['label'])

    ax.set_title(title, pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, None)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_jaccard_line(df: pd.DataFrame,
                      title: str = "Jaccard Index by Config",
                      save_path: Optional[Path] = None,
                      condition_order: Optional[List[str]] = None,
                      config_order: Optional[List[str]] = None,
                      extra_series: Optional[dict] = None) -> None:
    """
    Creates a line plot showing Jaccard index trends across configs for each condition.
    """
    plot_metric_line(df, metric_col='jaccard_index', title=title,
                     ylabel='Jaccard Index', save_path=save_path,
                     condition_order=condition_order, config_order=config_order,
                     extra_series=extra_series)


def plot_metric_vs_probability_scatter(df: pd.DataFrame,
                                       metric_col: str,
                                       title: str,
                                       xlabel: str,
                                       save_path: Optional[Path] = None,
                                       condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a scatter plot showing relationship between a metric and output probability.
    Points are colored by condition with regression lines for each.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()

    fig, ax = plt.subplots(figsize=(9, 6))

    palette = CUSTOM_PALETTE[:len(condition_order)]

    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition]
        ax.scatter(cond_df[metric_col], cond_df['output_probability'],
                   color=palette[i], label=condition, alpha=0.7, s=60, edgecolor='white')

        # Add regression line
        if len(cond_df) > 1:
            z = np.polyfit(cond_df[metric_col], cond_df['output_probability'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(cond_df[metric_col].min(), cond_df[metric_col].max(), 100)
            ax.plot(x_line, p(x_line), color=palette[i], linestyle='--', alpha=0.7, linewidth=1.5)

    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel('Output Probability', labelpad=10)
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_metric_vs_probability_combined(df: pd.DataFrame,
                                        title: str = "Jaccard Metrics vs Output Probability",
                                        save_path: Optional[Path] = None,
                                        condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a 2x1 subplot showing both Jaccard and Weighted Jaccard vs output probability.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = CUSTOM_PALETTE[:len(condition_order)]

    for ax, metric_col, xlabel in zip(axes, ['jaccard_index', 'relative_jaccard'],
                                      ['Jaccard Index', 'Weighted Jaccard']):
        for i, condition in enumerate(condition_order):
            cond_df = df[df['prompt_type'] == condition]
            ax.scatter(cond_df[metric_col], cond_df['output_probability'],
                       color=palette[i], label=condition, alpha=0.7, s=60, edgecolor='white')

            if len(cond_df) > 1:
                z = np.polyfit(cond_df[metric_col], cond_df['output_probability'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(cond_df[metric_col].min(), cond_df[metric_col].max(), 100)
                ax.plot(x_line, p(x_line), color=palette[i], linestyle='--', alpha=0.7, linewidth=1.5)

        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel('Output Probability', labelpad=10)
        sns.despine(ax=ax, left=True, bottom=True)

    axes[0].legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_probability_by_condition(df: pd.DataFrame,
                                  title: str = "Output Probability by Condition",
                                  save_path: Optional[Path] = None,
                                  condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a boxplot showing output probability distribution by condition.
    """
    plot_metric_boxplot(df, metric_col='output_probability', title=title,
                        ylabel='Output Probability', save_path=save_path,
                        condition_order=condition_order)


def plot_correlation_heatmap(df: pd.DataFrame,
                             title: str = "Metric Correlations by Condition",
                             save_path: Optional[Path] = None,
                             condition_order: Optional[List[str]] = None) -> None:
    """
    Creates a heatmap showing correlations between metrics for each condition.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()

    metrics = ['jaccard_index', 'relative_jaccard', 'output_probability']
    metric_labels = ['Jaccard', 'Weighted Jaccard', 'Output Prob']

    n_conditions = len(condition_order)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5 * n_conditions, 4))

    if n_conditions == 1:
        axes = [axes]

    for ax, condition in zip(axes, condition_order):
        cond_df = df[df['prompt_type'] == condition][metrics]
        corr = cond_df.corr()

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, linewidths=0.5, linecolor='white',
                    xticklabels=metric_labels, yticklabels=metric_labels,
                    ax=ax, cbar=ax == axes[-1])

        ax.set_title(condition, fontsize=11, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def analyze_overlap(dir1: Path, dir2: Path,
                    save_dir: Optional[Path] = None,
                    condition_order: Optional[List[str]] = None,
                    config_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Main function to load CSVs, combine them, and generate all visualizations.
    """
    df = load_and_combine_csvs(dir1, dir2, filename=OVERLAP_ANALYSIS_FILENAME)

    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())
    if config_order is None:
        config_order = sorted(df['config_name'].unique().tolist())

    # Check for error-results-individual.csv in dir1 to add average_precision line
    ap_series = None
    error_results_path = Path(dir1) / "memorized vs. random" / "error-results-individual.csv"
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
        print(f"Saved overlap analysis results to: {save_dir}")
    else:
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

    return df


def run_for_config(config_dir: Path, config_name: str,
                   run_error_analysis: bool, submodel_num: int = 0) -> tuple[Optional[dict], Optional[dict[str, dict]]]:
    """
    Runs the main logic for a specified prompt config.
    """
    config_path = config_dir / f"{config_name}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

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
    if selected_task == Task.PROMPT_SUBGRAPH_COMPARE:
        print(f"\nStarting run for {config_name} with model {model} and submodel {submodel}")
        print("\n".join([f"{p_id}: {p}" if p_id != p else p for p, p_id in prompt2ids.items()]))
        print("==================================")

        metrics = compare_prompt_subgraphs(main_prompt=prompt, diff_prompts=diff_prompts, sim_prompts=sim_prompts,
                                           model=model, submodel=submodel, graph_dir=graph_dir, filter_by_act_density=None,
                                           debug=False)
        if metrics:
            metrics = {prompt2ids.get(p, p): res for p, res in metrics.items()}
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
    else:
        raise NotImplementedError("Unknown task")
    return metrics, error_analysis_results


def run_for_all_configs(config_names: List[str] = None, config_dir: str = None,
                        run_error_analysis: bool = False, submodel_num: int = 0):
    """
    Runs analysis across all configs.
    """
    assert config_names or config_dir, "Must provide config names or config dir!"
    config_dir = Path(config_dir)
    if config_dir:
        config_dir = Path(CONFIG_BASE_DIR) / config_dir
    if not config_names or config_names[0].lower().strip() == 'all':
        config_names = [f.stem for f in config_dir.glob("*.json")]
    all_results = {"config_name": [], "prompt_type": [],
                   **{metric_name: [] for metric_name in IntersectionMetrics._fields}}
    error_pair_results = {}
    for config in config_names:
        config = config.strip()
        results, error_analysis_results = run_for_config(config_dir, config, submodel_num=submodel_num,
                                                         run_error_analysis=run_error_analysis)
        results: dict[str, IntersectionMetrics]
        if error_analysis_results:
            error_pair_results[config] = error_analysis_results
        if results:
            for i, (prompt, metrics) in enumerate(results.items()):
                all_results["config_name"].append(config)
                all_results["prompt_type"].append(prompt)
                for metric_name, metric in zip(IntersectionMetrics._fields, metrics):
                    all_results[metric_name].append(metric)

    results_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(OUTPUT_DIR) / results_dir
    os.makedirs(save_path, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(save_path / OVERLAP_ANALYSIS_FILENAME)

    if error_pair_results:
        analyze_conditions(error_pair_results, save_path=save_path)
