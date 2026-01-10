import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from src.constants import OUTPUT_DIR, CONFIG_BASE_DIR, FEATURE_LAYER, FEATURE_ID, PROMPT_IDS_MEMORIZED
from src.visualizations import visualize_feature_presence

load_dotenv()


def visualize_feature_presence_from_file(results_dir: Path):
    """Visualize feature presence counts by prompt_id."""
    csv_path = results_dir / f"feature_{FEATURE_LAYER}_{FEATURE_ID}.csv"
    df = pd.read_csv(csv_path)

    visualize_feature_presence(df, results_dir)

    # Visualize relationship between feature presence and output_prob if available
    if 'output_prob' in df.columns:
        visualize_feature_vs_output_prob(df, results_dir)

    return df


def visualize_feature_vs_output_prob(df: pd.DataFrame, results_dir: Path):
    """Visualize relationship between feature presence and output probability."""
    # Exclude rows without output_prob
    df = df.dropna(subset=['output_prob'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot: output_prob by feature_present
    ax1 = axes[0]
    present_probs = df[df['feature_present'] == True]['output_prob']
    absent_probs = df[df['feature_present'] == False]['output_prob']
    ax1.boxplot([present_probs, absent_probs], labels=['Present', 'Absent'])
    ax1.set_xlabel('Feature Status')
    ax1.set_ylabel('Output Probability')
    ax1.set_title(f'Output Probability by Feature {FEATURE_LAYER}/{FEATURE_ID} Presence')

    # Scatter plot with jitter
    ax2 = axes[1]
    jitter = 0.1
    present_x = [0 + random.uniform(-jitter, jitter) for _ in range(len(present_probs))]
    absent_x = [1 + random.uniform(-jitter, jitter) for _ in range(len(absent_probs))]
    ax2.scatter(present_x, present_probs, alpha=0.6, color='#2ecc71', label='Present')
    ax2.scatter(absent_x, absent_probs, alpha=0.6, color='#e74c3c', label='Absent')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Present', 'Absent'])
    ax2.set_xlabel('Feature Status')
    ax2.set_ylabel('Output Probability')
    ax2.set_title(f'Output Probability vs Feature {FEATURE_LAYER}/{FEATURE_ID} Presence')
    ax2.legend()

    plt.tight_layout()

    # Save
    save_path = results_dir / f"feature_{FEATURE_LAYER}_{FEATURE_ID}_vs_output_prob.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")


def create_weird_config(df: pd.DataFrame, config_dir: str):
    """Create configs/testing/weird.json with memorized prompts that have the feature."""
    # Filter for memorized prompts with feature present

    # Load actual prompts from config files
    feature_present = []
    feature_not_present = []
    for row in df.itertuples():
        config_name = row.config_name
        config_path = Path(CONFIG_BASE_DIR) / config_dir / f"{config_name}.json"
        with open(config_path) as f:
            config = json.load(f)
        prompt_id = row.prompt_id
        if prompt_id == "mem" or prompt_id == "memorized":
            prompt = config['MAIN_PROMPT']
        else:
            index = PROMPT_IDS_MEMORIZED.index(prompt_id)
            prompt = config['DIFF_PROMPTS'][index-1]
        if row.feature_present:
            feature_present.append(prompt)
        else:
            feature_not_present.append(prompt)

    if not feature_present:
        print("No memorized prompts with feature found")
        return

    # Pick random one as MAIN_PROMPT, rest as DIFF_PROMPTS
    main_prompt = random.choice(feature_present)
    diff_prompts = [p for p in feature_present if p != main_prompt]
    sim_prompts = [p for p in feature_not_present if p != main_prompt]

    # Create the config
    weird_config = {
        "MAIN_PROMPT": main_prompt,
        "DIFF_PROMPTS": diff_prompts,
        "SIM_PROMPTS": sim_prompts
    }

    # Save to configs/testing/weird.json
    output_path = Path(CONFIG_BASE_DIR) / "testing" / "weird.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weird_config, f, indent=2)

    print(f"Created {output_path} with {len(diff_prompts)} prompts")


if __name__ == "__main__":
    results_dir = os.getenv("RESULTS_DIR", "")
    config_dir = os.getenv("CONFIG_DIR", "overlap")
    reload_path = Path(OUTPUT_DIR) / results_dir
    df = visualize_feature_presence_from_file(reload_path)
    # create_weird_config(df, config_dir)
