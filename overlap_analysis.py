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
    OUTPUT_DIR, OVERLAP_ANALYSIS_FILENAME, COLORS, CUSTOM_PALETTE, FEATURE_LAYER, FEATURE_ID
from error_comparisons import run_error_ranking, analyze_conditions
from subgraph_comparisons import (compare_prompt_subgraphs, compare_token_subgraphs, compare_prompts_combined,
                                  IntersectionMetrics, CombinedMetrics, SharedFeatureMetrics)
from visualizations import visualize_feature_presence

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


def plot_combined_metrics(df: pd.DataFrame,
                          save_dir: Optional[Path] = None,
                          condition_order: Optional[List[str]] = None,
                          config_order: Optional[List[str]] = None) -> None:
    """
    Creates all visualizations for combined overlap metrics (unique/shared fractions).
    """
    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())
    if config_order is None:
        config_order = sorted(df['config_name'].unique().tolist())

    palette = CUSTOM_PALETTE[:len(condition_order)]

    # Line plot for unique_frac by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name')
        cond_df = cond_df.reindex(config_order)
        ax.plot(config_order, cond_df['unique_frac'],
                marker='o', markersize=8, linewidth=2,
                color=palette[i], label=condition)

    ax.set_title('Unique Feature Fraction by Config', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction Unique to Main', labelpad=10)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "combined_unique_frac_line.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Line plot for shared_frac by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name')
        cond_df = cond_df.reindex(config_order)
        ax.plot(config_order, cond_df['shared_frac'],
                marker='o', markersize=8, linewidth=2,
                color=palette[i], label=condition)

    ax.set_title('Shared Feature Fraction by Config', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction Shared Among All', labelpad=10)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "combined_shared_frac_line.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Line plot for weighted fractions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name')
        cond_df = cond_df.reindex(config_order)
        ax.plot(config_order, cond_df['unique_weighted_frac'],
                marker='o', markersize=8, linewidth=2,
                color=palette[i], label=condition)
    ax.set_title('Unique Weighted Fraction', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction', labelpad=10)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    ax = axes[1]
    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name')
        cond_df = cond_df.reindex(config_order)
        ax.plot(config_order, cond_df['shared_weighted_frac'],
                marker='o', markersize=8, linewidth=2,
                color=palette[i], label=condition)
    ax.set_title('Shared Weighted Fraction', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction', labelpad=10)
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "combined_weighted_frac_line.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Additional plots
    plot_combined_boxplot(df, save_path=save_dir / "combined_boxplot.png" if save_dir else None,
                          condition_order=condition_order)
    plot_combined_bar(df, save_path=save_dir / "combined_bar.png" if save_dir else None,
                      condition_order=condition_order, config_order=config_order)
    plot_combined_heatmap(df, save_path=save_dir / "combined_heatmap.png" if save_dir else None,
                          condition_order=condition_order, config_order=config_order)


def plot_combined_boxplot(df: pd.DataFrame,
                           save_path: Optional[Path] = None,
                           condition_order: Optional[List[str]] = None) -> None:
    """
    Creates boxplots comparing unique/shared fractions across conditions.
    """
    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unique fraction boxplot
    ax = axes[0]
    sns.boxplot(data=df, x='prompt_type', y='unique_frac', order=condition_order,
                palette=CUSTOM_PALETTE[:len(condition_order)], ax=ax)
    ax.set_title('Unique Feature Fraction by Condition', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Fraction Unique to Main', labelpad=10)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Shared fraction boxplot
    ax = axes[1]
    sns.boxplot(data=df, x='prompt_type', y='shared_frac', order=condition_order,
                palette=CUSTOM_PALETTE[:len(condition_order)], ax=ax)
    ax.set_title('Shared Feature Fraction by Condition', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Fraction Shared Among All', labelpad=10)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_combined_bar(df: pd.DataFrame,
                       save_path: Optional[Path] = None,
                       condition_order: Optional[List[str]] = None,
                       config_order: Optional[List[str]] = None) -> None:
    """
    Creates grouped bar charts comparing unique/shared fractions across conditions for each config.
    """
    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())
    if config_order is None:
        config_order = sorted(df['config_name'].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Unique fraction bar chart
    ax = axes[0]
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values='unique_frac')
    pivot_df = pivot_df.reindex(config_order)[condition_order]
    x = np.arange(len(config_order))
    width = 0.8 / len(condition_order)
    for i, condition in enumerate(condition_order):
        offset = (i - len(condition_order)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[condition], width,
               label=condition, color=CUSTOM_PALETTE[i])
    ax.set_title('Unique Feature Fraction by Config', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Shared fraction bar chart
    ax = axes[1]
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values='shared_frac')
    pivot_df = pivot_df.reindex(config_order)[condition_order]
    for i, condition in enumerate(condition_order):
        offset = (i - len(condition_order)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[condition], width,
               label=condition, color=CUSTOM_PALETTE[i])
    ax.set_title('Shared Feature Fraction by Config', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Fraction', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_combined_heatmap(df: pd.DataFrame,
                           save_path: Optional[Path] = None,
                           condition_order: Optional[List[str]] = None,
                           config_order: Optional[List[str]] = None) -> None:
    """
    Creates heatmaps for unique/shared fractions by config and condition.
    """
    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())
    if config_order is None:
        config_order = sorted(df['config_name'].unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Unique fraction heatmap
    ax = axes[0]
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values='unique_frac')
    pivot_df = pivot_df.reindex(config_order)[condition_order]
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Fraction'})
    ax.set_title('Unique Feature Fraction', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Config', labelpad=10)

    # Shared fraction heatmap
    ax = axes[1]
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values='shared_frac')
    pivot_df = pivot_df.reindex(config_order)[condition_order]
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Fraction'})
    ax.set_title('Shared Feature Fraction', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Config', labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_shared_feature_metrics(df: pd.DataFrame,
                                 save_dir: Optional[Path] = None,
                                 condition_order: Optional[List[str]] = None,
                                 config_order: Optional[List[str]] = None) -> None:
    """
    Creates visualizations for shared feature metrics across conditions.
    Includes threshold curve visualizations showing feature counts at different sharing thresholds.
    """
    if condition_order is None:
        condition_order = sorted(df['prompt_type'].unique().tolist())
    if config_order is None:
        config_order = sorted(df['config_name'].unique().tolist())

    palette = CUSTOM_PALETTE[:len(condition_order)]
    thresholds = [50, 75, 100]
    threshold_cols = ['count_at_50pct', 'count_at_75pct', 'count_at_100pct']
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # Threshold curve: feature counts at different thresholds (averaged across configs)
    # Use x-offsets, different line styles, and markers to distinguish overlapping lines
    fig, ax = plt.subplots(figsize=(10, 6))
    n_conditions = len(condition_order)
    x_offset_range = 1.5  # total offset range
    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition]
        means = [cond_df[col].mean() for col in threshold_cols]
        # Apply small x-offset to separate overlapping lines
        x_offset = (i - (n_conditions - 1) / 2) * (x_offset_range / max(n_conditions - 1, 1))
        x_positions = [t + x_offset for t in thresholds]
        ax.plot(x_positions, means,
                marker=markers[i % len(markers)], markersize=10, linewidth=2.5,
                linestyle=line_styles[i % len(line_styles)],
                color=palette[i], label=condition, alpha=0.85)

    ax.set_title('Feature Sharing Threshold Curve (Avg Across Configs)', pad=15)
    ax.set_xlabel('Sharing Threshold (%)', labelpad=10)
    ax.set_ylabel('Number of Features', labelpad=10)
    ax.set_xticks(thresholds)
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "shared_feature_threshold_curve.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Threshold curve per config (faceted)
    fig, axes = plt.subplots(1, len(config_order), figsize=(4 * len(config_order), 5), sharey=True)
    if len(config_order) == 1:
        axes = [axes]
    for ax, config in zip(axes, config_order):
        for i, condition in enumerate(condition_order):
            cond_df = df[(df['prompt_type'] == condition) & (df['config_name'] == config)]
            if len(cond_df) > 0:
                counts = [cond_df[col].values[0] for col in threshold_cols]
                ax.plot(thresholds, counts, marker='o', markersize=6, linewidth=2,
                        color=palette[i])
        ax.set_title(config, pad=10)
        ax.set_xlabel('Threshold (%)', labelpad=5)
        ax.set_xticks(thresholds)
    axes[0].set_ylabel('Number of Features', labelpad=10)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "shared_feature_threshold_curve_by_config.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Helper to parse comma-separated string to list of ints
    def parse_counts(s):
        if isinstance(s, str) and s:
            return [int(v) for v in s.split(',')]
        return []

    # Error bar plot: mean shared features present per prompt with std and range
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(config_order))
    width = 0.8 / len(condition_order)

    for i, condition in enumerate(condition_order):
        cond_df = df[df['prompt_type'] == condition].set_index('config_name').reindex(config_order)
        means, stds, mins, maxs = [], [], [], []
        for config in config_order:
            if config in cond_df.index:
                counts = parse_counts(cond_df.loc[config, 'shared_present_per_prompt'])
                if counts:
                    means.append(np.mean(counts))
                    stds.append(np.std(counts))
                    mins.append(min(counts))
                    maxs.append(max(counts))
                else:
                    means.append(0)
                    stds.append(0)
                    mins.append(0)
                    maxs.append(0)
            else:
                means.append(0)
                stds.append(0)
                mins.append(0)
                maxs.append(0)

        offset = (i - len(condition_order)/2 + 0.5) * width
        # Plot bars with error bars showing std
        ax.bar(x + offset, means, width, label=condition, color=palette[i], alpha=0.8)
        ax.errorbar(x + offset, means, yerr=stds, fmt='none', color='black', capsize=3, capthick=1)
        # Add range indicators (min-max) as thin lines
        for j, (xi, mn, mx) in enumerate(zip(x + offset, mins, maxs)):
            ax.plot([xi, xi], [mn, mx], color='black', linewidth=1, alpha=0.5)

    ax.set_title('Shared Features Present per Prompt (Mean ± Std, with Min-Max Range)', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Count of Shared Features in Each Prompt', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "shared_present_per_prompt.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Boxplot: shared features present per prompt across conditions
    boxplot_data = []
    for _, row in df.iterrows():
        counts = parse_counts(row['shared_present_per_prompt'])
        for val in counts:
            boxplot_data.append({
                'prompt_type': row['prompt_type'],
                'config_name': row['config_name'],
                'shared_present': val
            })
    boxplot_df = pd.DataFrame(boxplot_data)

    if not boxplot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=boxplot_df, x='prompt_type', y='shared_present', order=condition_order,
                    palette=palette, ax=ax)
        sns.stripplot(data=boxplot_df, x='prompt_type', y='shared_present', order=condition_order,
                      color='#333333', alpha=0.4, size=4, jitter=0.2, ax=ax)
        ax.set_title('Shared Features Present per Prompt', pad=15)
        ax.set_xlabel('Condition', labelpad=10)
        ax.set_ylabel('Count of Shared Features in Each Prompt', labelpad=10)
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / "shared_present_per_prompt_boxplot.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()

        # Bar chart: mean shared features per prompt by condition with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        condition_means = boxplot_df.groupby('prompt_type')['shared_present'].mean().reindex(condition_order)
        condition_stds = boxplot_df.groupby('prompt_type')['shared_present'].std().reindex(condition_order)

        bars = ax.bar(range(len(condition_order)), condition_means.values,
                      color=[palette[i] for i in range(len(condition_order))], alpha=0.8)
        ax.errorbar(range(len(condition_order)), condition_means.values, yerr=condition_stds.values,
                    fmt='none', color='black', capsize=5, capthick=1.5)

        ax.set_title('Mean Shared Features per Prompt by Condition', pad=15)
        ax.set_xlabel('Condition', labelpad=10)
        ax.set_ylabel('Mean Shared Features per Prompt', labelpad=10)
        ax.set_xticks(range(len(condition_order)))
        ax.set_xticklabels(condition_order, rotation=45, ha='right')
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / "shared_present_mean_bar.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()

    # Bar chart: num_shared / avg_features_per_prompt by condition
    df['shared_ratio'] = df['num_shared'] / df['avg_features_per_prompt']
    ratio_means = df.groupby('prompt_type')['shared_ratio'].mean().reindex(condition_order)
    ratio_stds = df.groupby('prompt_type')['shared_ratio'].std().reindex(condition_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(condition_order)), ratio_means.values,
                  color=[palette[i] for i in range(len(condition_order))], alpha=0.8)
    ax.errorbar(range(len(condition_order)), ratio_means.values, yerr=ratio_stds.values,
                fmt='none', color='black', capsize=5, capthick=1.5)

    ax.set_title('Shared Features Ratio by Condition', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Num Shared / Avg Features per Prompt', labelpad=10)
    ax.set_xticks(range(len(condition_order)))
    ax.set_xticklabels(condition_order, rotation=45, ha='right')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "shared_ratio_bar.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()

    # Bar chart comparing num_shared (at 50% threshold) by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_df = df.pivot(index='config_name', columns='prompt_type', values='num_shared')
    pivot_df = pivot_df.reindex(config_order)[condition_order]
    x = np.arange(len(config_order))
    width = 0.8 / len(condition_order)
    for i, condition in enumerate(condition_order):
        offset = (i - len(condition_order)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[condition], width,
               label=condition, color=palette[i])
    ax.set_title('Number of Shared Features by Config (50% threshold)', pad=15)
    ax.set_xlabel('Config', labelpad=10)
    ax.set_ylabel('Number of Shared Features', labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=45, ha='right')
    ax.legend(title='Condition', frameon=True, fancybox=True, shadow=True)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "shared_feature_num_bar.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


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
