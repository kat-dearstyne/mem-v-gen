from pathlib import Path
from typing import List, Optional, Dict

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from narwhals import DataFrame

from src.constants import FEATURE_LAYER, FEATURE_ID, CUSTOM_PALETTE, COLORS
from src.utils import get_conditions_from_label

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


def get_palette(n: int) -> List[str]:
    """
    Returns a list of n colors from the custom palette, cycling if n exceeds palette size.

    Args:
        n: Number of colors needed.

    Returns:
        List of hex color strings.
    """
    return [CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i in range(n)]


def visualize_feature_presence(df: DataFrame, results_dir: Path):
    """
    Creates a bar chart showing feature presence counts by prompt_id.

    Args:
        df: DataFrame with prompt_id and feature_present columns.
        results_dir: Directory to save the visualization.

    Returns:
        The input DataFrame unchanged.
    """

    # Count True/False for each prompt_id
    counts = df.groupby('prompt_id')['feature_present'].agg(['sum', 'count'])
    counts.columns = ['Present', 'Total']
    counts['Absent'] = counts['Total'] - counts['Present']
    counts = counts[['Present', 'Absent']]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])

    ax.set_xlabel('Prompt ID')
    ax.set_ylabel('Count')
    ax.set_title(f'Feature {FEATURE_LAYER}/{FEATURE_ID} Presence by Prompt Type')
    ax.legend(title='Feature Status')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save and show
    save_path = Path(results_dir) / f"feature_{FEATURE_LAYER}_{FEATURE_ID}_viz.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")

    return df


def plot_metric_by_condition(df: pd.DataFrame,
                             metric_col: str,
                             title: str,
                             ylabel: str,
                             save_path: Optional[Path] = None,
                             condition_order: Optional[List[str]] = None,
                             config_order: Optional[List[str]] = None) -> None:
    """
    Creates a grouped bar chart comparing a metric across conditions for each config.

    Args:
        df: DataFrame with config_name, prompt_type, and metric columns.
        metric_col: Column name for the metric to plot.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
        config_order: Optional ordering for configs.
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

    palette = get_palette(n_conditions)

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
    Creates a grouped bar chart comparing Jaccard index across conditions.

    Args:
        df: DataFrame with jaccard_index, config_name, and prompt_type columns.
        title: Plot title.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
        config_order: Optional ordering for configs.
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

    Args:
        df: DataFrame with config_name, prompt_type, and metric columns.
        metric_col: Column name for the metric to plot.
        title: Plot title.
        cbar_label: Colorbar label.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
        config_order: Optional ordering for configs.
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

    Args:
        df: DataFrame with jaccard_index, config_name, and prompt_type columns.
        title: Plot title.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
        config_order: Optional ordering for configs.
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

    Args:
        df: DataFrame with prompt_type and metric columns.
        metric_col: Column name for the metric to plot.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    palette = get_palette(len(condition_order))

    sns.boxplot(x='prompt_type', y=metric_col, data=df, hue='prompt_type',
                order=condition_order, hue_order=condition_order, palette=palette,
                width=0.5, linewidth=1.5, fliersize=0, legend=False, ax=ax)
    sns.stripplot(x='prompt_type', y=metric_col, data=df,
                  order=condition_order, color='#333333', alpha=0.6,
                  size=6, jitter=0.15, ax=ax)

    ax.set_title(title, pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.tick_params(axis='x', rotation=45)

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

    Args:
        df: DataFrame with jaccard_index and prompt_type columns.
        title: Plot title.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
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
        df: DataFrame with config_name, prompt_type, and metric columns.
        metric_col: Column name for the metric to plot.
        title: Plot title.
        ylabel: Y-axis label.
        save_path: Optional path to save the figure.
        condition_order: Optional ordering for conditions.
        config_order: Optional ordering for configs.
        extra_series: Optional dict with 'label' and 'data' keys for additional line.
    """
    if condition_order is None:
        condition_order = df['prompt_type'].unique().tolist()
    if config_order is None:
        config_order = df['config_name'].unique().tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    palette = get_palette(len(condition_order))

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

    palette = get_palette(len(condition_order))

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

    palette = get_palette(len(condition_order))

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

    palette = get_palette(len(condition_order))

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
    sns.boxplot(data=df, x='prompt_type', y='unique_frac', hue='prompt_type',
                order=condition_order, hue_order=condition_order,
                palette=get_palette(len(condition_order)), legend=False, ax=ax)
    ax.set_title('Unique Feature Fraction by Condition', pad=15)
    ax.set_xlabel('Condition', labelpad=10)
    ax.set_ylabel('Fraction Unique to Main', labelpad=10)
    ax.set_ylim(0, 1)
    sns.despine(ax=ax, left=True, bottom=True)

    # Shared fraction boxplot
    ax = axes[1]
    sns.boxplot(data=df, x='prompt_type', y='shared_frac', hue='prompt_type',
                order=condition_order, hue_order=condition_order,
                palette=get_palette(len(condition_order)), legend=False, ax=ax)
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
    palette = get_palette(len(condition_order))
    for i, condition in enumerate(condition_order):
        offset = (i - len(condition_order)/2 + 0.5) * width
        ax.bar(x + offset, pivot_df[condition], width,
               label=condition, color=palette[i])
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
               label=condition, color=palette[i])
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

    palette = get_palette(len(condition_order))
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
        sns.boxplot(data=boxplot_df, x='prompt_type', y='shared_present', hue='prompt_type',
                    order=condition_order, hue_order=condition_order,
                    palette=palette, legend=False, ax=ax)
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


def plot_error_hypothesis_bar_chart(df: pd.DataFrame,
                                     metric: str,
                                     title: str,
                                     conditions: List[str],
                                     palette: List,
                                     save_path: Path,
                                     is_bounded: bool = True) -> None:
    """
    Creates a bar chart with error bars for a metric across conditions.
    """
    ylabel = "Score" if is_bounded else "KL Divergence (nats)"

    fig, ax = plt.subplots(figsize=(8, 6))
    means = df.groupby("condition")[metric].mean().reindex(conditions)
    stds = df.groupby("condition")[metric].std().reindex(conditions)

    ax.bar(range(len(conditions)), means.values, color=palette, alpha=0.8)
    ax.errorbar(range(len(conditions)), means.values, yerr=stds.values,
                fmt='none', color='black', capsize=5)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    if is_bounded:
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_hypothesis_boxplot(df: pd.DataFrame,
                                   metric: str,
                                   title: str,
                                   conditions: List[str],
                                   palette: List,
                                   save_path: Path,
                                   is_bounded: bool = True) -> None:
    """
    Creates a boxplot with stripplot overlay for a metric across conditions.
    """
    ylabel = "Score" if is_bounded else "KL Divergence (nats)"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="condition", y=metric, hue="condition",
                order=conditions, hue_order=conditions,
                palette=palette, legend=False, ax=ax)
    sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                  color='black', alpha=0.5, size=4, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_hypothesis_heatmap(df: pd.DataFrame,
                                   metric: str,
                                   title: str,
                                   conditions: List[str],
                                   save_path: Path,
                                   is_bounded: bool = True) -> None:
    """
    Creates a heatmap of metric values across configs and conditions.
    """
    ylabel = "Score" if is_bounded else "KL Divergence (nats)"

    fig, ax = plt.subplots(figsize=(10, 8))
    pivot = df.pivot(index="config", columns="condition", values=metric)
    pivot = pivot[conditions]  # Reorder columns
    vmin, vmax = (0, 1) if is_bounded else (None, None)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label': ylabel})
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel("Config")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_hypothesis_combined_boxplot(df: pd.DataFrame,
                                            conditions: List[str],
                                            save_path: Path,
                                            metrics: Optional[List[str]] = None,
                                            titles: Optional[List[str]] = None) -> None:
    """
    Creates a single figure with boxplots for multiple metrics.

    Args:
        df: DataFrame with columns: condition and metric columns
        conditions: List of condition names
        save_path: Path to save the figure
        metrics: List of metric column names (default: 4 main metrics)
        titles: List of titles for each subplot (default: formatted metric names)
    """
    if metrics is None:
        metrics = ["last_token_cosine", "cumulative_cosine", "original_accuracy", "kl_divergence"]
    if titles is None:
        titles = ["Last Token Cosine", "Cumulative Cosine", "Original Accuracy", "KL Divergence"]

    palette = get_palette(len(conditions))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        is_bounded = metric != "kl_divergence"
        ylabel = "Score" if is_bounded else "KL Divergence (nats)"

        sns.boxplot(data=df, x="condition", y=metric, hue="condition",
                    order=conditions, hue_order=conditions,
                    palette=palette, legend=False, ax=ax)
        sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                      color='black', alpha=0.5, size=3, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_error_hypothesis_metrics(df: pd.DataFrame,
                                   output_dir: Path,
                                   top_k: int = 10) -> None:
    """
    Creates all visualizations for error hypothesis analysis.

    Args:
        df: DataFrame with columns: condition, config, and metric columns
        output_dir: Directory to save visualizations
        top_k: Number for top-k agreement metric label
    """
    conditions = df["condition"].unique().tolist()
    palette = get_palette(len(conditions))

    # All metrics with their display titles
    all_metrics = ["last_token_cosine", "cumulative_cosine", "original_accuracy",
                   "top_k_agreement", "replacement_prob_of_original_top", "kl_divergence"]
    all_titles = ["Last Token Cosine", "Cumulative Cosine", "Original Accuracy",
                  f"Top-{top_k} Agreement", "Replacement P(Original Top)", "KL Divergence"]

    # Create directories for each plot type
    bar_dir = output_dir / "bar_charts"
    boxplot_dir = output_dir / "boxplots"
    heatmap_dir = output_dir / "heatmaps"
    bar_dir.mkdir(parents=True, exist_ok=True)
    boxplot_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual plots for each metric
    for metric, title in zip(all_metrics, all_titles):
        is_bounded = metric != "kl_divergence"

        plot_error_hypothesis_bar_chart(
            df, metric, title, conditions, palette,
            bar_dir / f"{metric}.png", is_bounded
        )
        plot_error_hypothesis_boxplot(
            df, metric, title, conditions, palette,
            boxplot_dir / f"{metric}.png", is_bounded
        )
        plot_error_hypothesis_heatmap(
            df, metric, title, conditions,
            heatmap_dir / f"{metric}.png", is_bounded
        )


METRIC_LABELS = {
    "last_token_cosine": "Last Token Cosine",
    "cumulative_cosine": "Cumulative Cosine",
    "original_accuracy": "Original Accuracy",
    "kl_divergence": "KL Divergence",
    "top_k_agreement": "Top-10 Agreement",
    "replacement_prob_of_original_top": "Repl. P(Original Top)",
}

SIG_COLORS = {
    "both": "#06402B",      # Green - both tests significant
    "parametric_only": "#96D9C0",    # Light green - parametric only
    "none": "#bdc3c7",      # Gray - not significant
}


def _prepare_sig_plot_df(df: pd.DataFrame,
                          exclude_metrics: Optional[List[str]],
                          parametric_col: str,
                          nonparametric_col: str) -> pd.DataFrame:
    """Prepare DataFrame for significance plotting with labels and significance levels."""
    if exclude_metrics is None:
        exclude_metrics = []

    plot_df = df[~df["metric"].isin(exclude_metrics)].copy()
    if plot_df.empty:
        return plot_df

    plot_df["metric_label"] = plot_df["metric"].map(lambda m: METRIC_LABELS.get(m, m))

    def get_sig_level(row):
        if row[parametric_col] and row[nonparametric_col]:
            return "both"
        elif row[parametric_col]:
            return "parametric_only"
        return "none"

    plot_df["sig_level"] = plot_df.apply(get_sig_level, axis=1)
    return plot_df


def _plot_effect_size_bar(ax, plot_df: pd.DataFrame,
                           effect_col: str,
                           parametric_col: str,
                           nonparametric_col: str,
                           xlabel: str,
                           title: str,
                           thresholds: List[tuple]) -> None:
    """Plot a single effect size horizontal bar chart."""
    colors = [SIG_COLORS[level] for level in plot_df["sig_level"]]
    y_pos = range(len(plot_df))

    ax.barh(y_pos, plot_df[effect_col].abs(), color=colors, edgecolor='white', linewidth=0.8)

    # Add significance markers
    for i, (_, row) in enumerate(plot_df.iterrows()):
        marker = ""
        if row[parametric_col] and row[nonparametric_col]:
            marker = "**"
        elif row[parametric_col]:
            marker = "*"
        x_pos = abs(row[effect_col]) + 0.02
        ax.text(x_pos, i, marker, va='center', fontsize=12, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["metric_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='bold')

    # Add threshold lines
    for val, color in thresholds:
        ax.axvline(x=val, color=color, linestyle='--', alpha=0.5)

    ax.set_xlim(0, max(plot_df[effect_col].abs().max() * 1.2, thresholds[-1][0] * 1.5))
    sns.despine(ax=ax, left=True, bottom=True)


def _save_sig_figure(fig, save_path: Path, title: str, legend_labels: tuple) -> None:
    """Add legend and save significance figure."""
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SIG_COLORS["both"], label=f'Both tests sig. **'),
        Patch(facecolor=SIG_COLORS["parametric_only"], label=f'{legend_labels[0]} only sig. *'),
        Patch(facecolor=SIG_COLORS["none"], label='Not significant'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_significance_effect_sizes(df: pd.DataFrame,
                                    title: str,
                                    save_path: Path,
                                    exclude_metrics: Optional[List[str]] = None) -> None:
    """
    Plot pairwise significance with Cohen's d and rank-biserial r effect sizes.

    Uses BH-corrected p-values for significance determination.

    Args:
        df: DataFrame with columns: metric, t_significant_bh, mw_significant_bh, cohens_d, rank_biserial_r
        title: Title for the plot
        save_path: Path to save the visualization
        exclude_metrics: List of metrics to exclude
    """
    plot_df = _prepare_sig_plot_df(df, exclude_metrics, "t_significant_bh", "mw_significant_bh")
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_effect_size_bar(axes[0], plot_df, "cohens_d", "t_significant_bh", "mw_significant_bh",
                          "|Cohen's d|", "Effect Size: Cohen's d",
                          [(0.2, '#999'), (0.5, '#666'), (0.8, '#333')])

    _plot_effect_size_bar(axes[1], plot_df, "rank_biserial_r", "t_significant_bh", "mw_significant_bh",
                          "|Rank-biserial r|", "Effect Size: Rank-biserial r",
                          [(0.1, '#999'), (0.3, '#666'), (0.5, '#333')])

    _save_sig_figure(fig, save_path, title, ("T-test (BH)", "MW (BH)"))


def plot_omnibus_effect_sizes(df: pd.DataFrame,
                               title: str,
                               save_path: Path,
                               exclude_metrics: Optional[List[str]] = None) -> None:
    """
    Plot omnibus (ANOVA/Kruskal-Wallis) significance with eta² and epsilon² effect sizes.

    Args:
        df: DataFrame with columns: metric, anova_significant, kruskal_significant,
            eta_squared, epsilon_squared
        title: Title for the plot
        save_path: Path to save the visualization
        exclude_metrics: List of metrics to exclude
    """
    plot_df = _prepare_sig_plot_df(df, exclude_metrics, "anova_significant", "kruskal_significant")
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # eta² thresholds: 0.01 small, 0.06 medium, 0.14 large
    _plot_effect_size_bar(axes[0], plot_df, "eta_squared", "anova_significant", "kruskal_significant",
                          "η² (eta-squared)", "Effect Size: η² (ANOVA)",
                          [(0.01, '#999'), (0.06, '#666'), (0.14, '#333')])

    # epsilon² uses similar thresholds
    _plot_effect_size_bar(axes[1], plot_df, "epsilon_squared", "anova_significant", "kruskal_significant",
                          "ε² (epsilon-squared)", "Effect Size: ε² (Kruskal-Wallis)",
                          [(0.01, '#999'), (0.06, '#666'), (0.14, '#333')])

    _save_sig_figure(fig, save_path, title, ("ANOVA", "Kruskal"))


def plot_token_complexity(df: pd.DataFrame,
                          output_dir: Path,
                          conditions: List[str]) -> None:
    """
    Creates visualizations for token complexity analysis.

    Args:
        df: DataFrame with columns: condition, zipf_frequency, token_length
        output_dir: Directory to save visualizations (token_complexity subdir will be created)
        conditions: List of condition names
    """
    complexity_dir = output_dir / "token_complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)

    palette = get_palette(len(conditions))
    complexity_metrics = ["zipf_frequency", "token_length"]
    complexity_titles = ["Zipf Frequency (higher=more common)", "Token Length"]

    for metric, title in zip(complexity_metrics, complexity_titles):
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        means = df.groupby("condition")[metric].mean().reindex(conditions)
        stds = df.groupby("condition")[metric].std().reindex(conditions)

        ax.bar(range(len(conditions)), means.values, color=palette, alpha=0.8)
        ax.errorbar(range(len(conditions)), means.values, yerr=stds.values,
                    fmt='none', color='black', capsize=5)

        ax.set_title(f"Token Complexity: {title}", fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel(title)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(complexity_dir / f"{metric}_bar.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Boxplot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x="condition", y=metric, hue="condition",
                    order=conditions, hue_order=conditions,
                    palette=palette, legend=False, ax=ax)
        sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                      color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title(f"Token Complexity: {title}", fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(complexity_dir / f"{metric}_boxplot.png", dpi=150, bbox_inches='tight')
        plt.close()


def plot_per_position_curves(results: dict,
                              output_dir: Path,
                              conditions: List[str]) -> None:
    """
    Plot error metrics vs perplexity at each position for each prompt.

    Creates one plot per prompt (config), showing how error metrics relate to
    perplexity at each token position, with different conditions overlaid.

    Args:
        results: Dictionary structured as {config_name: {condition: metrics_dict}}
        output_dir: Directory to save visualizations
        conditions: List of condition names
    """
    curves_dir = output_dir / "per_position_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    palette = get_palette(len(conditions))
    condition_colors = {cond: palette[i] for i, cond in enumerate(conditions)}

    # Build configs_data from results
    configs_data = {}
    for config_name, condition_metrics in results.items():
        configs_data[config_name] = {}
        for condition, metrics in condition_metrics.items():
            configs_data[config_name][condition] = {
                "per_position_cosine": metrics.get("per_position_cosine", []),
                "per_position_kl": metrics.get("per_position_kl", []),
                "per_position_argmax_match": metrics.get("per_position_argmax_match", []),
                "per_position_cross_entropy": metrics.get("per_position_cross_entropy", []),
            }

    if not configs_data:
        return

    # Build list of all (config, condition) pairs to process
    pairs_to_process = [
        (config_name, condition)
        for config_name in configs_data
        for condition in conditions
        if condition in configs_data[config_name]
    ]

    # Plot each config (prompt) separately
    for config_name, condition in tqdm(pairs_to_process, desc="Plotting per-position curves"):
        cond_data = configs_data[config_name]
        safe_name = config_name.replace(" ", "_").replace("/", "_")

        cosine_values = cond_data[condition].get("per_position_cosine", [])
        ce_values = cond_data[condition].get("per_position_cross_entropy", [])

        if not cosine_values or not ce_values:
            continue

        # Convert cross-entropy to perplexity
        perplexity = [np.exp(v) for v in ce_values]

        # Ensure same length
        min_len = min(len(cosine_values), len(perplexity))
        cosine_values = cosine_values[:min_len]
        perplexity = perplexity[:min_len]
        positions = list(range(min_len))

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot cosine similarity on left y-axis
        color1 = '#1f77b4'
        ax1.plot(positions, cosine_values,
                 color=color1,
                 linewidth=2,
                 marker='o',
                 markersize=3,
                 label='Cosine Similarity')
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Cosine Similarity", color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Plot perplexity on right y-axis (log scale)
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.plot(positions, perplexity,
                 color=color2,
                 linewidth=2,
                 marker='s',
                 markersize=3,
                 label='Perplexity')
        ax2.set_ylabel("Perplexity (log scale)", color=color2)
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        ax1.set_title(f"{config_name} - {condition}", fontweight='bold')
        sns.despine(ax=ax1, right=False)

        plt.tight_layout()
        plt.savefig(curves_dir / f"{safe_name}_{condition}_cosine.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        # Plot KL divergence and perplexity
        kl_values = cond_data[condition].get("per_position_kl", [])
        if kl_values:
            min_len_kl = min(len(kl_values), len(perplexity))
            kl_values = kl_values[:min_len_kl]
            perplexity_kl = perplexity[:min_len_kl]
            positions_kl = list(range(min_len_kl))

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot KL divergence on left y-axis (log scale since KL can vary widely)
            color1 = '#2ca02c'
            ax1.plot(positions_kl, kl_values,
                     color=color1,
                     linewidth=2,
                     marker='o',
                     markersize=3,
                     label='KL Divergence')
            ax1.set_xlabel("Position")
            ax1.set_ylabel("KL Divergence", color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)

            # Plot perplexity on right y-axis (log scale)
            ax2 = ax1.twinx()
            color2 = '#ff7f0e'
            ax2.plot(positions_kl, perplexity_kl,
                     color=color2,
                     linewidth=2,
                     marker='s',
                     markersize=3,
                     label='Perplexity')
            ax2.set_ylabel("Perplexity (log scale)", color=color2)
            ax2.set_yscale('log')
            ax2.tick_params(axis='y', labelcolor=color2)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            ax1.set_title(f"{config_name} - {condition} (KL Divergence)", fontweight='bold')
            sns.despine(ax=ax1, right=False)

            plt.tight_layout()
            plt.savefig(curves_dir / f"{safe_name}_{condition}_kl.png", dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()


def plot_delta_distribution(delta_values: List[float], title: str, label: str,
                            save_path: Optional[Path] = None) -> None:
    """
    Creates a histogram plot showing the distribution of delta values.
    Visualizes the density distribution of metric differences between two conditions.

    Args:
        delta_values: List of delta values to plot.
        title: Plot title.
        label: Comparison label in format 'condition1 vs. condition2'.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create histogram with KDE overlay
    sns.histplot(delta_values, bins=15, kde=True, color=COLORS['turquoise'], alpha=0.7,
                 edgecolor='white', linewidth=0.8, ax=ax)

    # Add vertical line at zero for reference
    ax.axvline(x=0, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')

    # Add mean line
    mean_val = np.mean(delta_values)
    ax.axvline(x=mean_val, color=COLORS['pastel_orange'], linestyle='-',
               linewidth=2, alpha=0.9, label=f'Mean: {mean_val:.3f}')

    comparisons = get_conditions_from_label(label)
    ax.set_title(title, pad=15)
    ax.set_xlabel(f"Δ ({comparisons[0]} - {comparisons[1]})", labelpad=10)
    ax.set_ylabel("Density", labelpad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()


def plot_early_layer_boxplot(df: pd.DataFrame,
                              condition_order: List[str],
                              value_col: str = 'early_layer_fraction',
                              condition_col: str = 'prompt_type',
                              save_path: Optional[Path] = None) -> None:
    """
    Creates boxplot comparing early layer contribution between conditions.

    Args:
        df: DataFrame with early layer fractions.
        condition_order: Order of conditions for x-axis.
        value_col: Column name for the values.
        condition_col: Column name for the condition.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [df[df[condition_col] == cond][value_col].dropna().values
            for cond in condition_order]

    bp = ax.boxplot(data, labels=condition_order, patch_artist=True)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)])
        patch.set_alpha(0.7)

    ax.set_ylabel('Early Layer Contribution Fraction')
    ax.set_xlabel('Condition')
    ax.set_title('Early Layer Contribution by Condition')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_early_layer_mean_comparison(df: pd.DataFrame,
                                      condition_order: List[str],
                                      p_value: Optional[float] = None,
                                      value_col: str = 'early_layer_fraction',
                                      condition_col: str = 'prompt_type',
                                      save_path: Optional[Path] = None) -> None:
    """
    Creates bar chart with means and error bars for early layer contribution.

    Args:
        df: DataFrame with early layer fractions.
        condition_order: Order of conditions for x-axis.
        p_value: Optional p-value for significance annotation.
        value_col: Column name for the values.
        condition_col: Column name for the condition.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    stds = []
    for cond in condition_order:
        cond_data = df[df[condition_col] == cond][value_col].dropna()
        means.append(cond_data.mean())
        stds.append(cond_data.std())

    x = np.arange(len(condition_order))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i in range(len(condition_order))],
           alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(condition_order)
    ax.set_ylabel('Early Layer Contribution Fraction')
    ax.set_xlabel('Condition')
    ax.set_title('Mean Early Layer Contribution by Condition')

    # Add significance annotation if p-value provided
    if p_value is not None and len(condition_order) == 2:
        sig_text = _get_significance_stars(p_value)
        if sig_text:
            y_max = max(means) + max(stds) * 1.5
            ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=1)
            ax.text(0.5, y_max * 1.02, sig_text, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_early_layer_by_config(df: pd.DataFrame,
                                condition_order: List[str],
                                config_order: List[str],
                                value_col: str = 'early_layer_fraction',
                                condition_col: str = 'prompt_type',
                                config_col: str = 'config_name',
                                save_path: Optional[Path] = None) -> None:
    """
    Creates line plot showing early layer contribution by config for each condition.

    Args:
        df: DataFrame with early layer fractions.
        condition_order: Order of conditions for legend.
        config_order: Order of configs for x-axis.
        value_col: Column name for the values.
        condition_col: Column name for the condition.
        config_col: Column name for the config.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cond in enumerate(condition_order):
        cond_df = df[df[condition_col] == cond]
        values = []
        for config in config_order:
            config_val = cond_df[cond_df[config_col] == config][value_col]
            values.append(config_val.values[0] if len(config_val) > 0 else np.nan)

        ax.plot(config_order, values, marker='o', label=cond,
                color=CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)], linewidth=2, markersize=8)

    ax.set_xlabel('Config')
    ax.set_ylabel('Early Layer Contribution Fraction')
    ax.set_title('Early Layer Contribution by Config and Condition')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _get_significance_stars(p_value: float) -> str:
    """
    Returns significance stars based on p-value.

    Args:
        p_value: The p-value from statistical test.

    Returns:
        String with asterisks indicating significance level.
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


def boxplot_metric_family(metric_dict: Dict[int, List[float]], title_prefix: str, label: str,
                          save_path: Optional[Path] = None) -> None:
    """
    Creates boxplot visualizations for a group of related metrics across different K values.
    Useful for visualizing NDCG@k or top-k error proportion across different k values.

    Args:
        metric_dict: Dictionary mapping K values to lists of delta values.
        title_prefix: Prefix for the plot title.
        label: Comparison label in format 'condition1 vs. condition2'.
        save_path: Optional path to save the figure.
    """
    ks = sorted(metric_dict.keys())

    # Prepare data in long format for seaborn
    plot_data = []
    for k in ks:
        for val in metric_dict[k]:
            plot_data.append({'K': f'K={k}', 'Delta': val})
    df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create boxplot with strip plot overlay
    n_k_values = df['K'].nunique()
    k_order = df['K'].unique().tolist()
    sns.boxplot(x='K', y='Delta', data=df, hue='K',
                order=k_order, hue_order=k_order,
                palette=get_palette(n_k_values), width=0.5,
                linewidth=1.5, fliersize=0, legend=False, ax=ax)
    sns.stripplot(x='K', y='Delta', data=df, color='#333333', alpha=0.5,
                  size=4, jitter=0.15, ax=ax)

    # Add horizontal line at zero
    ax.axhline(y=0, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title(f"{title_prefix}: Δ Distributions", pad=15)
    ax.set_xlabel("Top-K Value", labelpad=10)
    comparisons = get_conditions_from_label(label)
    ax.set_ylabel(f"Δ ({comparisons[0]} - {comparisons[1]})", labelpad=10)
    ax.tick_params(axis='x', rotation=45)

    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
