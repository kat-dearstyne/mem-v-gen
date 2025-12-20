#!/usr/bin/env python3
"""
Temporary script to generate markdown documentation for config files.
Reads all .json configs in the base configs dir and creates a markdown file
with prompts and their shared_token information from overlap-metrics.csv.
"""

import json
from pathlib import Path
import pandas as pd

CONFIG_DIR = Path("../configs")
OVERLAP_METRICS_PATH = Path("../output/memorized-clt/overlap-metrics.csv")
OUTPUT_PATH = Path("../data/config_prompts.md")

PROMPT_TYPES = ["memorized", "rephrased", "made-up", "random"]
PROMPT_TYPE_TO_CSV_SUFFIX = {
    "rephrased": "memorized vs rephrased",
    "made-up": "memorized vs made-up",
    "random": "memorized vs. random",  # Note the period in "vs."
}


def main():
    # Load overlap metrics CSV
    df = pd.read_csv(OVERLAP_METRICS_PATH)

    # Get all json files in base configs dir (not nested)
    config_files = sorted(CONFIG_DIR.glob("*.json"))

    lines = ["# Config Prompts\n"]

    for config_path in config_files:
        config_name = config_path.stem

        with open(config_path, "r") as f:
            config = json.load(f)

        lines.append(f"## {config_name}\n")

        # Get the memorized prompt
        main_prompt = config.get("MAIN_PROMPT", "")
        diff_prompts = config.get("DIFF_PROMPTS", [])

        # Build list of (prompt_type, prompt)
        prompts = [("memorized", main_prompt)]
        for i, prompt in enumerate(diff_prompts):
            # DIFF_PROMPTS order: rephrased, made-up, random
            prompt_type = PROMPT_TYPES[i + 1] if i + 1 < len(PROMPT_TYPES) else f"diff_{i}"
            prompts.append((prompt_type, prompt))

        # Get shared_token from CSV for this config (we'll extract the base info from any row)
        config_rows = df[df["config_name"] == config_name]

        # Get base shared_token info (without the p value) from first row
        base_shared_token = ""
        if len(config_rows) > 0:
            sample_token = config_rows.iloc[0]["shared_token"]
            # Extract everything except the p value: 'Output " dog" (p=0.043)' -> 'Output " dog"'
            if "(p=" in sample_token:
                base_shared_token = sample_token.split("(p=")[0].strip()

        for prompt_type, prompt in prompts:
            if prompt_type == "memorized":
                # Memorized prompt is not in overlap-metrics, use placeholder
                shared_token = f"{base_shared_token} (p=[TODO])" if base_shared_token else "[TODO]"
            else:
                # Look up in CSV
                csv_prompt_type = PROMPT_TYPE_TO_CSV_SUFFIX.get(prompt_type, "")
                row = config_rows[config_rows["prompt_type"] == csv_prompt_type]
                if len(row) > 0:
                    shared_token = row.iloc[0]["shared_token"]
                else:
                    shared_token = "[NOT FOUND]"

            lines.append(f"- **{prompt_type}:** {prompt}")
            lines.append(f"  - {shared_token}")

        lines.append("")  # Empty line between configs

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
