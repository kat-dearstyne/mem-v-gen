"""
Visualization script for L0 analysis results from Gemma Scope 2 CLTs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from src.constants import OUTPUT_DIR

# Style settings
CUSTOM_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_l0_bar_chart(
    df: pd.DataFrame,
    layer_col: str = "layer",
    l0_col: str = "l0_value",
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Creates a bar chart showing mean L0 per layer with error bars.

    Args:
        df: DataFrame with layer and l0_value columns.
        layer_col: Column name for layer.
        l0_col: Column name for L0 values.
        title: Optional title for the plot.
        save_path: Optional path to save the figure.
    """
    is_normalized = "normalized" in l0_col.lower()
    y_label = "L0 / Total Features" if is_normalized else "L0 (Active Features)"
    default_title = "Normalized L0 Per Layer" if is_normalized else "L0 Per Layer"
    title = title or default_title

    # Compute statistics per layer
    stats = df.groupby(layer_col)[l0_col].agg(['mean', 'std']).reset_index()
    layers = stats[layer_col].values
    means = stats['mean'].values
    stds = stats['std'].fillna(0).values

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(layers, means, yerr=stds, capsize=3,
                  color=CUSTOM_PALETTE[0], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Layer", fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(layers)

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved bar chart to: {save_path}")

    plt.show()
    plt.close()


def plot_l0_line_chart(
    df: pd.DataFrame,
    layer_col: str = "layer",
    l0_col: str = "l0_value",
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Creates a line plot showing mean L0 per layer with shaded std region.

    Args:
        df: DataFrame with layer and l0_value columns.
        layer_col: Column name for layer.
        l0_col: Column name for L0 values.
        title: Optional title for the plot.
        save_path: Optional path to save the figure.
    """
    is_normalized = "normalized" in l0_col.lower()
    y_label = "L0 / Total Features" if is_normalized else "L0 (Active Features)"
    default_title = "Normalized L0 Per Layer" if is_normalized else "L0 Per Layer"
    title = title or default_title

    # Compute statistics per layer
    stats = df.groupby(layer_col)[l0_col].agg(['mean', 'std']).reset_index()
    layers = stats[layer_col].values
    means = stats['mean'].values
    stds = stats['std'].fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 6))

    color = CUSTOM_PALETTE[0]
    ax.plot(layers, means, marker='o', markersize=6, linewidth=2,
            color=color, label='Mean L0')
    ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.2,
                    label='±1 Std Dev')

    ax.set_xlabel("Layer", fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(layers)
    ax.legend(loc='best')

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved line chart to: {save_path}")

    plt.show()
    plt.close()


def plot_l0_boxplot(
    df: pd.DataFrame,
    layer_col: str = "layer",
    l0_col: str = "l0_value",
    title: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Creates a boxplot showing L0 distribution per layer.

    Args:
        df: DataFrame with layer and l0_value columns.
        layer_col: Column name for layer.
        l0_col: Column name for L0 values.
        title: Optional title for the plot.
        save_path: Optional path to save the figure.
    """
    is_normalized = "normalized" in l0_col.lower()
    y_label = "L0 / Total Features" if is_normalized else "L0 (Active Features)"
    default_title = "L0 Distribution Per Layer" if not is_normalized else "Normalized L0 Distribution Per Layer"
    title = title or default_title

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df, x=layer_col, y=l0_col, ax=ax,
                color=CUSTOM_PALETTE[0], fliersize=3)

    ax.set_xlabel("Layer", fontsize=12, labelpad=10)
    ax.set_ylabel(y_label, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')

    sns.despine(ax=ax)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved boxplot to: {save_path}")

    plt.show()
    plt.close()


def visualize_l0_results(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    normalized: bool = False
) -> None:
    """
    Generate all L0 visualizations from saved results.

    Args:
        data_dir: Directory containing l0_per_layer.csv and l0_stats.csv.
        output_dir: Directory to save plots (defaults to data_dir).
        normalized: Whether to plot normalized L0 values.
    """
    output_dir = output_dir or data_dir

    # Load data
    df = pd.read_csv(data_dir / "l0_per_layer.csv")
    stats_df = pd.read_csv(data_dir / "l0_stats.csv")

    print(f"Loaded {len(df)} rows from l0_per_layer.csv")
    print(f"Layers: {df['layer'].nunique()}, Prompts: {df['prompt_id'].nunique()}")

    l0_col = "l0_normalized" if normalized else "l0_value"
    suffix = "_normalized" if normalized else ""

    # Generate plots
    plot_l0_bar_chart(
        df, l0_col=l0_col,
        title="Gemma Scope 2 CLT - L0 Per Layer",
        save_path=output_dir / f"l0_bar_chart{suffix}.png"
    )

    plot_l0_line_chart(
        df, l0_col=l0_col,
        title="Gemma Scope 2 CLT - L0 Per Layer",
        save_path=output_dir / f"l0_line_chart{suffix}.png"
    )

    plot_l0_boxplot(
        df, l0_col=l0_col,
        title="Gemma Scope 2 CLT - L0 Distribution Per Layer",
        save_path=output_dir / f"l0_boxplot{suffix}.png"
    )

    # Print summary statistics
    print("\nSummary Statistics (from l0_stats.csv):")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    import sys

    # Default path
    data_dir = Path(f"{OUTPUT_DIR}/l0_gemmascope2")

    # Allow command line override
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Generate both raw and normalized visualizations
    visualize_l0_results(data_dir, normalized=False)
    visualize_l0_results(data_dir, normalized=True)
