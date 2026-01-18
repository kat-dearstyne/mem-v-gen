import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Union

from src.utils import load_json, get_env_list


class Task:
    PROMPT_SUBGRAPH_COMPARE = "prompt"
    TOKEN_SUBGRAPH_COMPARE = "token"
    FEATURE_OVERLAP = "feature_overlap"
    REPLACEMENT_MODEL = "replacement_model"
    EARLY_LAYER_CONTRIBUTION = "early_layer"
    L0_REPLACEMENT_MODEL = "l0"


@dataclass
class DatasetConfig:
    """Configuration for loading prompts from a HuggingFace dataset."""

    name: str
    num_samples: int
    text_column: str = "text"
    split: str = "train"
    subset: Optional[str] = None
    seed: int = 42


@dataclass
class AnalysisConfig:
    """Configuration for running analysis on a prompt config."""

    main_prompt: str
    prompt_ids: Dict[str, str]
    diff_prompts: List[str] = field(default_factory=list)
    sim_prompts: List[str] = field(default_factory=list)
    token_of_interest: Optional[str] = None
    task: Optional[str] = None
    memorized_completion: Optional[str] = None
    dataset: Optional[DatasetConfig] = None

    @classmethod
    def from_file(cls, config_path: Path, prompt_ids_override: List[str] = None) -> "AnalysisConfig":
        """
        Load an AnalysisConfig from a JSON file.

        Args:
            config_path: Path to the JSON config file.
            prompt_ids_override: Optional list of prompt IDs to use instead of config/env.

        Returns:
            AnalysisConfig instance with values from file and environment.
        """
        config_data = load_json(config_path)

        # Check for dataset config
        dataset_data = config_data.get("DATASET")
        if dataset_data:
            return cls._from_dataset_config(config_data, dataset_data)

        main_prompt = config_data["MAIN_PROMPT"]
        diff_prompts = config_data.get("DIFF_PROMPTS", [])
        sim_prompts = config_data.get("SIM_PROMPTS", [])
        token_of_interest = config_data.get("TOKEN_OF_INTEREST")
        memorized_completion = config_data.get("MEMORIZED_COMPLETION")

        # Parse prompt IDs: override > config > environment
        all_prompts = [main_prompt] + diff_prompts + sim_prompts
        prompt_ids = cls._parse_prompt_ids(
            config_data.get("PROMPT_IDS") or prompt_ids_override,
            all_prompts
        )

        # Get task from config and environment, config takes priority
        config_task = config_data.get("TASK")
        env_task = os.getenv("TASK", "").strip() or None

        if config_task and env_task and config_task != env_task:
            print(f"Warning: TASK mismatch - config has '{config_task}', "
                  f"environment has '{env_task}'. Using config value.")

        task = config_task or env_task

        # Infer task from other fields if not specified
        if not task:
            if diff_prompts or sim_prompts:
                if token_of_interest:
                    raise ValueError(
                        "Both TOKEN_OF_INTEREST and DIFF_PROMPTS/SIM_PROMPTS supplied. "
                        "Must specify TASK to disambiguate."
                    )
                task = Task.PROMPT_SUBGRAPH_COMPARE
            elif token_of_interest:
                task = Task.TOKEN_SUBGRAPH_COMPARE

        return cls(
            main_prompt=main_prompt,
            diff_prompts=diff_prompts,
            sim_prompts=sim_prompts,
            token_of_interest=token_of_interest,
            task=task,
            memorized_completion=memorized_completion,
            prompt_ids=prompt_ids,
        )

    @classmethod
    def _from_dataset_config(cls, config_data: dict, dataset_data: dict) -> "AnalysisConfig":
        """
        Create an AnalysisConfig from a dataset configuration.

        Args:
            config_data: Full config data dict.
            dataset_data: Dataset configuration dict with 'name', 'num_samples', etc.

        Returns:
            AnalysisConfig instance configured for dataset-based L0 analysis.
        """
        dataset = DatasetConfig(
            name=dataset_data["name"],
            num_samples=dataset_data["num_samples"],
            text_column=dataset_data.get("text_column", "text"),
            split=dataset_data.get("split", "train"),
            subset=dataset_data.get("subset"),
            seed=dataset_data.get("seed", 42),
        )

        return cls(
            main_prompt="",
            prompt_ids={},
            task=config_data.get("TASK", Task.L0_REPLACEMENT_MODEL),
            dataset=dataset,
        )

    @staticmethod
    def _parse_prompt_ids(
            prompt_ids: Optional[Union[List[str], Dict[str, List[str]]]],
            all_prompts: List[str]
    ) -> Dict[str, str]:
        """
        Parse PROMPT_IDS from config or environment into a prompt -> ID mapping.

        Args:
            prompt_ids: Either a list of IDs or a dict with 'main', 'diff', 'sim' keys.
            all_prompts: All prompts in order [main, *diff, *sim].

        Returns:
            Dictionary mapping prompt string to its ID.
        """
        # Try config first, then environment variable
        if prompt_ids is None:
            prompt_ids = get_env_list("PROMPT_IDS")
            if not prompt_ids:
                raise ValueError(
                    "PROMPT_IDS must be specified in config file or PROMPT_IDS environment variable"
                )

        # Convert dict format to list format
        if isinstance(prompt_ids, dict):
            prompt_ids = (
                    prompt_ids.get("main", []) +
                    prompt_ids.get("diff", []) +
                    prompt_ids.get("sim", [])
            )

        # Validate and build mapping from list format
        if len(prompt_ids) != len(all_prompts):
            raise ValueError(
                f"PROMPT_IDS length ({len(prompt_ids)}) must match "
                f"total prompts ({len(all_prompts)})"
            )

        return {prompt: pid for prompt, pid in zip(all_prompts, prompt_ids)}

    @property
    def all_prompts(self) -> List[str]:
        """Returns all prompts (main + diff + sim)."""
        return [self.main_prompt] + self.diff_prompts + self.sim_prompts

    @property
    def id_to_prompt(self) -> Dict[str, str]:
        """Returns mapping from prompt ID to prompt string."""
        return {pid: prompt for prompt, pid in self.prompt_ids.items()}

    def get_prompt_id(self, prompt: str) -> str:
        """Get the ID for a given prompt string."""
        return self.prompt_ids[prompt]

    def get_prompt(self, prompt_id: str) -> str:
        """Get the prompt string for a given ID."""
        return self.id_to_prompt[prompt_id]

    @property
    def main_prompt_id(self) -> str:
        """Get the ID for the main prompt."""
        return self.prompt_ids[self.main_prompt]

    @property
    def diff_prompt_ids(self) -> List[str]:
        """Get the IDs for diff prompts."""
        return [self.prompt_ids[p] for p in self.diff_prompts]

    @property
    def sim_prompt_ids(self) -> List[str]:
        """Get the IDs for sim prompts."""
        return [self.prompt_ids[p] for p in self.sim_prompts]
