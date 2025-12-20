from pathlib import Path

from matplotlib import pyplot as plt
from narwhals import DataFrame

from constants import FEATURE_LAYER, FEATURE_ID


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
