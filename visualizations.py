from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from narwhals import DataFrame

from constants import FEATURE_LAYER, FEATURE_ID, CUSTOM_PALETTE

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


def visualize_feature_presence(df: DataFrame, results_dir: Path):
    """Visualize feature presence counts by prompt_id."""

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
    sns.boxplot(data=df, x="condition", y=metric, order=conditions,
                palette=palette, ax=ax)
    sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                  color='black', alpha=0.5, size=4, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(conditions, rotation=45, ha='right')

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
                                            palette: List,
                                            save_path: Path,
                                            metrics: Optional[List[str]] = None,
                                            titles: Optional[List[str]] = None) -> None:
    """
    Creates a single figure with boxplots for multiple metrics.

    Args:
        df: DataFrame with columns: condition and metric columns
        conditions: List of condition names
        palette: Color palette for conditions
        save_path: Path to save the figure
        metrics: List of metric column names (default: 4 main metrics)
        titles: List of titles for each subplot (default: formatted metric names)
    """
    if metrics is None:
        metrics = ["last_token_cosine", "cumulative_cosine", "original_accuracy", "kl_divergence"]
    if titles is None:
        titles = ["Last Token Cosine", "Cumulative Cosine", "Original Accuracy", "KL Divergence"]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        is_bounded = metric != "kl_divergence"
        ylabel = "Score" if is_bounded else "KL Divergence (nats)"

        sns.boxplot(data=df, x="condition", y=metric, order=conditions,
                    palette=palette, ax=ax)
        sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                      color='black', alpha=0.5, size=3, ax=ax)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)

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
    palette = sns.color_palette("husl", len(conditions))

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


def plot_significance_effect_sizes(df: pd.DataFrame,
                                    title: str,
                                    save_path: Path,
                                    exclude_metrics: Optional[List[str]] = None) -> None:
    """
    Creates a visualization showing significance and effect sizes for metrics.

    Displays a horizontal bar chart with Cohen's d effect sizes, with significance
    indicated by bar color and asterisks.

    Args:
        df: DataFrame with columns: metric, t_significant, mw_significant, cohens_d, rank_biserial_r
        title: Title for the plot
        save_path: Path to save the visualization
        exclude_metrics: List of metrics to exclude from visualization
    """
    if exclude_metrics is None:
        exclude_metrics = []

    # Filter out excluded metrics
    plot_df = df[~df["metric"].isin(exclude_metrics)].copy()

    if plot_df.empty:
        return

    # Create readable metric labels
    metric_labels = {
        "last_token_cosine": "Last Token Cosine",
        "cumulative_cosine": "Cumulative Cosine",
        "original_accuracy": "Original Accuracy",
        "kl_divergence": "KL Divergence",
        "top_k_agreement": "Top-10 Agreement",
        "replacement_prob_of_original_top": "Repl. P(Original Top)",
    }
    plot_df["metric_label"] = plot_df["metric"].map(lambda m: metric_labels.get(m, m))

    # Determine significance level for coloring
    # Both significant = dark green, t-test only = light green, neither = gray
    def get_sig_level(row):
        if row["t_significant"] and row["mw_significant"]:
            return "both"
        elif row["t_significant"]:
            return "t_only"
        else:
            return "none"

    plot_df["sig_level"] = plot_df.apply(get_sig_level, axis=1)

    # Color mapping
    sig_colors = {
        "both": "#06402B",      # Green - both tests significant
        "t_only": "#96D9C0",    # Light green - t-test only
        "none": "#bdc3c7",      # Gray - not significant
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Cohen's d
    ax = axes[0]
    colors = [sig_colors[level] for level in plot_df["sig_level"]]
    y_pos = range(len(plot_df))

    bars = ax.barh(y_pos, plot_df["cohens_d"].abs(), color=colors, edgecolor='white', linewidth=0.8)

    # Add significance markers
    for i, (_, row) in enumerate(plot_df.iterrows()):
        marker = ""
        if row["t_significant"] and row["mw_significant"]:
            marker = "**"
        elif row["t_significant"]:
            marker = "*"

        x_pos = abs(row["cohens_d"]) + 0.02
        ax.text(x_pos, i, marker, va='center', fontsize=12, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["metric_label"])
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Effect Size: Cohen's d", fontweight='bold')
    ax.axvline(x=0.2, color='#999', linestyle='--', alpha=0.5, label='small (0.2)')
    ax.axvline(x=0.5, color='#666', linestyle='--', alpha=0.5, label='medium (0.5)')
    ax.axvline(x=0.8, color='#333', linestyle='--', alpha=0.5, label='large (0.8)')
    ax.set_xlim(0, max(plot_df["cohens_d"].abs().max() * 1.2, 1.2))
    sns.despine(ax=ax, left=True, bottom=True)

    # Plot Rank-biserial r
    ax = axes[1]
    bars = ax.barh(y_pos, plot_df["rank_biserial_r"].abs(), color=colors, edgecolor='white', linewidth=0.8)

    # Add significance markers
    for i, (_, row) in enumerate(plot_df.iterrows()):
        marker = ""
        if row["t_significant"] and row["mw_significant"]:
            marker = "**"
        elif row["t_significant"]:
            marker = "*"

        x_pos = abs(row["rank_biserial_r"]) + 0.02
        ax.text(x_pos, i, marker, va='center', fontsize=12, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["metric_label"])
    ax.set_xlabel("|Rank-biserial r|")
    ax.set_title("Effect Size: Rank-biserial r", fontweight='bold')
    ax.axvline(x=0.1, color='#999', linestyle='--', alpha=0.5, label='small (0.1)')
    ax.axvline(x=0.3, color='#666', linestyle='--', alpha=0.5, label='medium (0.3)')
    ax.axvline(x=0.5, color='#333', linestyle='--', alpha=0.5, label='large (0.5)')
    ax.set_xlim(0, max(plot_df["rank_biserial_r"].abs().max() * 1.2, 0.7))
    sns.despine(ax=ax, left=True, bottom=True)

    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=sig_colors["both"], label='Both tests sig. **'),
        Patch(facecolor=sig_colors["t_only"], label='T-test only sig. *'),
        Patch(facecolor=sig_colors["none"], label='Not significant'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_token_complexity(df: pd.DataFrame,
                          output_dir: Path,
                          conditions: List[str],
                          palette: List) -> None:
    """
    Creates visualizations for token complexity analysis.

    Args:
        df: DataFrame with columns: condition, zipf_frequency, token_length
        output_dir: Directory to save visualizations (token_complexity subdir will be created)
        conditions: List of condition names
        palette: Color palette for conditions
    """
    complexity_dir = output_dir / "token_complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)

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
        sns.boxplot(data=df, x="condition", y=metric, order=conditions,
                    palette=palette, ax=ax)
        sns.stripplot(data=df, x="condition", y=metric, order=conditions,
                      color='black', alpha=0.5, size=4, ax=ax)
        ax.set_title(f"Token Complexity: {title}", fontweight='bold')
        ax.set_xlabel("Condition")
        ax.set_ylabel(title)
        ax.set_xticklabels(conditions, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(complexity_dir / f"{metric}_boxplot.png", dpi=150, bbox_inches='tight')
        plt.close()
