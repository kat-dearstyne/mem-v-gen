import json
import random
from pathlib import Path

from constants import CONFIG_BASE_DIR


def create_memorized_and_nonmemorized_configs():
    """Create memorized.json and non-memorized.json in configs/testing."""
    base_configs_dir = Path(CONFIG_BASE_DIR)
    testing_dir = base_configs_dir / "testing"
    testing_dir.mkdir(parents=True, exist_ok=True)

    memorized_prompts = []
    non_memorized_prompts = []

    # Collect prompts from all json files in base configs dir
    for config_path in base_configs_dir.glob("*.json"):
        with open(config_path) as f:
            config = json.load(f)

        # Get MAIN_PROMPT for memorized
        if "MAIN_PROMPT" in config:
            memorized_prompts.append(config["MAIN_PROMPT"])

        # Get last item in DIFF_PROMPTS for non-memorized
        if "DIFF_PROMPTS" in config and config["DIFF_PROMPTS"]:
            non_memorized_prompts.append(config["DIFF_PROMPTS"][1])

    # Create memorized.json
    if memorized_prompts:
        main_prompt = random.choice(memorized_prompts)
        sim_prompts = [p for p in memorized_prompts if p != main_prompt]
        memorized_config = {
            "MAIN_PROMPT": main_prompt,
            "SIM_PROMPTS": sim_prompts
        }
        with open(testing_dir / "memorized.json", "w") as f:
            json.dump(memorized_config, f, indent=2)
        print(f"Created memorized.json with {len(memorized_prompts)} prompts")

    # Create non-memorized.json
    if non_memorized_prompts:
        main_prompt = random.choice(non_memorized_prompts)
        sim_prompts = [p for p in non_memorized_prompts if p != main_prompt]
        non_memorized_config = {
            "MAIN_PROMPT": main_prompt,
            "SIM_PROMPTS": sim_prompts
        }
        with open(testing_dir / "rephrased.json", "w") as f:
            json.dump(non_memorized_config, f, indent=2)
        print(f"Created non-memorized.json with {len(non_memorized_prompts)} prompts")


if __name__ == "__main__":
    create_memorized_and_nonmemorized_configs()
