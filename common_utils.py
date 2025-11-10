import _curses
import glob
import json
import os
import re
import uuid
from collections import namedtuple
from typing import Tuple, NamedTuple

import pandas as pd
from pick import pick

from constants import NEURONPEDIA_API_KEY_ENV_VAR, AVAILABLE_MODELS
Feature = namedtuple("Feature", ("layer", "feature"))

def save_df(df: pd.DataFrame, foldername: str, filename: str) -> str:
    """
    Save a DataFrame to a CSV file.
    """
    os.makedirs(foldername, exist_ok=True)
    save_path = os.path.join(foldername, filename)
    df.to_csv(save_path, index=False)
    return save_path


def save_json(json_dict: dict, save_path: str) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as json_file:
        json.dump(json_dict, json_file, indent=4)


def load_json(save_path: str) -> dict:
    """
    Load a dictionary from a JSON file.
    """
    with open(save_path, "r") as json_file:
        json_dict = json.load(json_file)
    return json_dict


def get_api_key() -> str:
    """
    Retrieve the Neuronpedia API key from environment variables.
    """
    assert os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR), f"Must set {NEURONPEDIA_API_KEY_ENV_VAR} in .env"
    return os.environ.get(NEURONPEDIA_API_KEY_ENV_VAR)


def get_most_recent_file(directory_path: str, pattern: str = "*") -> str | None:
    """
    Get the most recently modified file in a directory matching a pattern, searching subdirectories if needed.
    """
    full_pattern = os.path.join(directory_path, pattern)
    list_of_files = glob.glob(full_pattern)

    # Filter to only actual files (not directories)
    list_of_files = [f for f in list_of_files if os.path.isfile(f)]

    # If no files found and directory contains only subdirectories, search recursively
    if not list_of_files and os.path.exists(directory_path):
        dir_contents = os.listdir(directory_path)
        # Check if directory has contents and all are directories
        if dir_contents and all(os.path.isdir(os.path.join(directory_path, item)) for item in dir_contents):
            full_pattern = os.path.join(directory_path, "**", pattern)
            list_of_files = glob.glob(full_pattern, recursive=True)
            list_of_files = [f for f in list_of_files if os.path.isfile(f)]

    if not list_of_files:
        return None

    # Sort files by their modification time (getmtime) in ascending order
    # The last element after sorting will be the most recent
    list_of_files.sort(key=os.path.getmtime, reverse=True)
    return list_of_files[-1]


def user_select_prompt(prompt_default: str = None, graph_dir: str | None = None) -> str:
    """
    Allows the user to enter a prompt (or select the last one used) if a default isn't given.
    """
    latest_graph_file = get_most_recent_file(graph_dir) if graph_dir else None
    last_prompt = None
    if latest_graph_file:
        graph_metadata = load_json(latest_graph_file)
        last_prompt = graph_metadata["metadata"]["prompt"].replace("<bos>", "")
        print(f"MOST RECENT PROMPT: {last_prompt}")

    input_addition = " or skip to use last prompt" if last_prompt else ""
    prompt = input(f"Enter prompt{input_addition}:\n").strip() if not prompt_default else prompt_default
    prompt = prompt or last_prompt
    assert prompt, "No prompt given"
    return prompt


def user_select_models(model: str | None = None, submodel: str | None = None) -> Tuple[str, str]:
    """
    Allows the user to select a model and submodel if a default isn't given.
    """
    if not model:
        model, _ = pick(list(AVAILABLE_MODELS.keys()), "Select a model:")
    model = model.lower()
    submodels = AVAILABLE_MODELS[model]
    if not submodel or submodel not in submodels:
        try:
            submodel, _ = pick(submodels, "Select a submodel:") if len(submodels) > 1 else submodels[0]
        except _curses.error:
            submodel = submodels[0]
    submodel = submodel.lower()
    return model, submodel


def create_prompt_id(prompt: str) -> str:
    """
    Creates a unique prompt id for a specified prompt.
    """
    prompt_start = "_".join(prompt.lower().split()[:5])
    prompt_id = f"{prompt_start}:{uuid.uuid5(uuid.NAMESPACE_DNS, prompt)}"
    return prompt_id


def get_top_k_from_df(df: pd.DataFrame, k: int, sort_by: str | list[str], ascending: bool | list[bool] = False) -> pd.DataFrame:
    """
    Creates a df with the top k rows after sorting by given columns.
    """
    if isinstance(sort_by, list) and not isinstance(ascending, list):
        ascending = [ascending for _ in sort_by]
    sorted_df = df.sort_values(by=sort_by, ascending=ascending)
    top_df = sorted_df.head(min(k, len(sorted_df)))
    return top_df

def get_node_ids_from_features(feature_df: pd.DataFrame) -> list[str]:
    """
    Creates a list of each node id from a dataframe of features.
    """
    return [f"{feature.layer}-{feature.feature}" for feature in feature_df.itertuples()]

def create_node_id(feature: Feature, deliminator: str = "-") -> str:
    """
    Creates the node id in the format layer-feature where - is changed via deliminator.
    """
    return f"{feature.layer}{deliminator}{feature.feature}"


def get_feature_from_node_id(node_id: str, deliminator: str = "-") -> Feature:
    """
    Splits the node id into layer, feature based on given deliminator.
    """
    split_node_id = node_id.split(deliminator)
    return Feature(split_node_id[0], split_node_id[1])

def get_output_token_from_clerp(output_node: dict) -> str:
    """
    Gets the output token from an output node's clerp attribute.
    """
    match = re.search(r'"(.*?)"', output_node["clerp"])
    if match:
        token = match.group(1).strip()
        if not token and output_node["clerp"].count("\"") > 2:
            token = '\"'
        return token
    return ''