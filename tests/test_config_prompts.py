import json
import unittest
from pathlib import Path

CONFIG_BASE_DIR = Path(__file__).parent.parent / "configs"


def collect_prompts_from_config(config_path: Path) -> set:
    """Collects all prompts from a config file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Unable to load {config_path}")
        raise e

    prompts = set()

    if "MAIN_PROMPT" in config:
        prompts.add(config["MAIN_PROMPT"])

    if "DIFF_PROMPTS" in config:
        prompts.update(config["DIFF_PROMPTS"])

    if "SIM_PROMPTS" in config:
        prompts.update(config["SIM_PROMPTS"])

    return prompts


class TestConfigPrompts(unittest.TestCase):
    """Tests for config prompt consistency."""

    def test_unique_prompts_count(self):
        """
        Tests that:
        1. Each base config contains exactly 4 unique prompts
        2. Each baseline config contains exactly 3 unique prompts
        3. All 3 prompts in baseline configs also appear in the corresponding base config
        """
        base_config_dir = CONFIG_BASE_DIR
        baseline_config_dir = CONFIG_BASE_DIR / "baseline"

        # Get all json files in base configs dir (not subdirectories)
        base_config_files = {f.stem: f for f in base_config_dir.glob("*.json")}
        baseline_config_files = {f.stem: f for f in baseline_config_dir.glob("*.json")}

        self.assertGreater(len(base_config_files), 0, "No config files found in base configs directory")

        # Check each base config has exactly 4 prompts
        for config_name, config_path in base_config_files.items():
            base_prompts = collect_prompts_from_config(config_path)
            self.assertEqual(
                len(base_prompts), 4,
                f"Base config '{config_name}' should have 4 unique prompts, found {len(base_prompts)}"
            )

        # Check each baseline config has exactly 3 prompts and all appear in base config
        for config_name, config_path in baseline_config_files.items():
            baseline_prompts = collect_prompts_from_config(config_path)
            self.assertEqual(
                len(baseline_prompts), 3,
                f"Baseline config '{config_name}' should have 3 unique prompts, found {len(baseline_prompts)}"
            )

            # Check corresponding base config exists
            self.assertIn(
                config_name, base_config_files,
                f"Baseline config '{config_name}' has no corresponding base config"
            )

            # Check all baseline prompts appear in base config
            base_prompts = collect_prompts_from_config(base_config_files[config_name])
            missing_prompts = baseline_prompts - base_prompts
            self.assertEqual(
                len(missing_prompts), 0,
                f"Baseline config '{config_name}' contains prompts not in base config: {missing_prompts}"
            )
