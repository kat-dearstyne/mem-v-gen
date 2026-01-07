import json
import os
from enum import Enum
from pathlib import Path
from typing import Tuple

import pandas as pd


class Metrics(Enum):
    """Base enum class for metric definitions with display formatting."""

    def get_printable(self) -> str:
        """
        Returns the name with spaces instead of underscores.

        Returns:
            The metric name formatted with spaces and title case.
        """
        return " ".join(self.value.split("_")).title()


def save_df(df: pd.DataFrame, foldername: str, filename: str) -> str:
    """
    Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        foldername: Directory to save the file in (created if doesn't exist).
        filename: Name of the CSV file.

    Returns:
        Full path to the saved file.
    """
    os.makedirs(foldername, exist_ok=True)
    save_path = os.path.join(foldername, filename)
    df.to_csv(save_path, index=False)
    return save_path


def save_json(json_dict: dict, save_path: str | Path) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.

    Args:
        json_dict: Dictionary to save.
        save_path: Path to save the JSON file (parent directories created if needed).
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)


def load_json(save_path: str | Path) -> dict:
    """
    Load a dictionary from a JSON file.

    Args:
        save_path: Path to the JSON file to load.

    Returns:
        Dictionary loaded from the JSON file.
    """
    save_path = Path(save_path) if isinstance(save_path, str) else save_path
    with open(save_path, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict


def get_api_key() -> str:
    """
    Retrieve the Neuronpedia API key from environment variables.

    Returns:
        The API key string.

    Raises:
        AssertionError: If the environment variable is not set.
    """
    NEURONPEDIA_API_KEY_ENV_VAR = "NEURONPEDIA_API_KEY"
    assert os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR), f"Must set {NEURONPEDIA_API_KEY_ENV_VAR} in .env"
    return os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR)


def append_to_dict_list(d: dict, key, value) -> None:
    """
    Append a value to a list in a dictionary, creating the list if the key doesn't exist.

    Args:
        d: Dictionary to modify in place.
        key: Key for the list to append to.
        value: Value to append to the list.
    """
    if key not in d:
        d[key] = []
    d[key].append(value)


def create_label_from_conditions(condition1: str, condition2: str) -> str:
    """
    Creates a label describing comparison of condition1 to condition2.

    Args:
        condition1: The first condition.
        condition2: The second condition.

    Returns:
        Label in format 'condition1 vs. condition2'.
    """
    return f"{condition1} vs. {condition2}"


def get_conditions_from_label(label: str) -> Tuple[str, str]:
    """
    Gets the conditions being compared from a comparison label.

    Args:
        label: Label in format 'condition1 vs. condition2'.

    Returns:
        Tuple of (condition1, condition2).

    Raises:
        AssertionError: If the label doesn't contain exactly two conditions.
    """
    conditions = [cond.strip() for cond in label.split("vs.")]
    assert len(conditions) == 2, f"Cannot get conditions from label: {conditions}"
    return tuple(conditions)


def get_as_safe_name(orig_name: str) -> str:
    """
    Converts name to a safe name for saving as a file (-> snakecase).
    Args:
        orig_name: Name with spaces or dashes.

    Returns: Name as a safe name for saving as a file (-> snakecase).
    """
    return orig_name.replace(" ", "_").replace("-", "_")
