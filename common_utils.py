import _curses
import glob
import json
import os
import re
import uuid
from collections import namedtuple
from typing import Tuple, NamedTuple, Any, List, Set, Dict

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

def get_id_without_pos(node_name: str) -> Any:
    """
    Removes the position part of the id
    """
    if node_name.count("_") == 2:
        return node_name.rsplit("_", 1)[0]
    return node_name


def get_top_output_logit_node(nodes: List[dict]) -> Any:
    """
    Gets the the node that represents the target output logit from a list of all graph nodes.
    """
    return [node for node in nodes if node['is_target_logit']][0]


def get_output_logits(nodes: List[dict]) -> Any:
    """
    Gets the the node that represents the target output logit from a list of all graph nodes.
    """
    return [node for node in nodes if node['feature_type'] == "logit"]


def get_linked_sources(graph_metadata: dict, output_token_to_features: dict, positive_only: bool = True) -> bool:
    """
    Gets all sources linked to each output token and updates the output_token_to_features dictionary in place.
    Returns True if any new sources were added, False otherwise.
    """
    newly_added = False
    for link in graph_metadata["links"]:
        for token, nodes in output_token_to_features.items():
            if link["target"] in nodes and (not positive_only or link["weight"] > 0):
                if link["source"] not in nodes:
                    nodes.add(link["source"])
                    newly_added = True
    return newly_added


def get_links_from_node(graph_metadata: dict, starting_node: dict = None,
                        positive_only: bool = True, hops: int = None, include_features_only: bool = False):
    """
    Finds all links that are connected to a starting node with the option to include only positive edges,
    include edges for only n hops away from start, or include only edges connecting to 'cross layer transcoder' features.
    """
    allowed_nodes = {node["node_id"] for node in graph_metadata["nodes"]
                     if not include_features_only or node["feature_type"] == 'cross layer transcoder'}
    if starting_node is None:
        starting_node = get_top_output_logit_node(graph_metadata['nodes'])
    starting_node_id = starting_node["node_id"]
    target_id_to_links = {}
    for link in graph_metadata["links"]:
        target_id = link["target"]
        if target_id not in target_id_to_links:
            target_id_to_links[target_id] = []
        if not positive_only or link["weight"] > 0:
            target_id_to_links[target_id].append(link)
    relevant_links = []
    seen = set()
    new_targets = get_links_to_targets([starting_node_id], target_id_to_links, relevant_links, allowed_nodes)
    n_hops = 0
    while new_targets:
        n_hops += 1
        if hops and n_hops >= hops:
            break
        seen.update(new_targets)
        new_targets = get_links_to_targets(new_targets, target_id_to_links, relevant_links, allowed_nodes)
        new_targets = new_targets.difference(seen)

    return relevant_links


def get_nodes_linked_to_target(graph_metadata: dict, target_node: dict, should_sort: bool = True) -> List[dict]:
    """
    Gets all nodes that are directly linked to the target_node, with options to sort them by weight.
    Returns a list of node dictionaries sorted by link weight (descending) if should_sort is True.
    """
    all_nodes = get_node_dict(graph_metadata)
    links = get_links_from_node(graph_metadata, target_node, hops=1)
    if should_sort:
        links = sorted(links, key=lambda link: link['weight'], reverse=True)
    return [all_nodes[link['source']] for link in links]


def get_node_dict(graph_metadata: dict) -> dict[Any, Any]:
    """
    Gets a dictionary mapping node_id to node for each node of the graph.
    """
    node_dict = {node["node_id"]: node for node in graph_metadata['nodes']}
    return node_dict


def get_links_to_targets(target_ids: List | Set, target_id_to_links: Dict, links: list, allowed_nodes: set):
    """
    Get links to specified target ids.
    """
    for target_id in target_ids:
        linked = target_id_to_links.get(target_id, [])
        links.extend([link for link in linked if link["source"] in allowed_nodes])
    new_targets = {link["source"] for link in links}
    return new_targets


def get_overlap_scores_for_features(prompt_tokens: list[str], features: list[dict], tok_k_activations: int = 10) -> \
        list[int]:
    """
    Calculates the max overlap of activating tokens with the prompt for each feature.
    """

    def process_token(token: str):
        return token.lstrip('▁').lstrip().lower()

    prompt_tokens_set = {process_token(token) for token in prompt_tokens}
    overlap_scores = []
    for feature in features:
        feature_overlap_scores = []
        for act in feature["activations"][:tok_k_activations]:
            activating_tokens = {process_token(token) for token, val in zip(act["tokens"], act["values"]) if val > 0}
            overlap_score = len(prompt_tokens_set.intersection(activating_tokens))
            feature_overlap_scores.append(overlap_score)
        overlap_scores.append(max(feature_overlap_scores))
    return overlap_scores


def get_frequencies(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency counts for features.
    """
    node_df_clts_only = node_df[node_df["feature_type"] == "cross layer transcoder"]
    ctx_freq_df = node_df_clts_only.value_counts(["layer", "feature"]).reset_index(name="ctx_freq")
    return ctx_freq_df


def create_node_df(graph_metadata: dict, include_errors_by_pos: bool = False,
                   exclude_embeddings: bool = False, exclude_errors: bool = False,
                   exclude_logits: bool = False) -> pd.DataFrame:
    """
    Extract node information from graph metadata into a DataFrame.
    """
    feature_list = []
    layer_list = []
    feature_type_list = []
    ctx_idx_list = []

    for node in graph_metadata["nodes"]:
        feature_type = node["feature_type"]
        if exclude_embeddings and feature_type == 'embedding':
            continue
        if exclude_errors and 'error' in feature_type:
            continue
        if exclude_logits and feature_type == 'logit':
            continue
        feature_type_list.append(feature_type)

        ctx_idx = node["ctx_idx"]
        ctx_idx_list.append(ctx_idx)

        feature = get_feature_from_node_id(node["node_id"], deliminator="_")[1]
        if 'error' in feature_type and include_errors_by_pos:
            feature = f"{feature}{ctx_idx}"
        feature_list.append(feature)

        layer = node["layer"]
        layer_list.append(layer)

    node_df = pd.DataFrame({
        "feature": feature_list,
        "layer": layer_list,
        "feature_type": feature_type_list,
        "ctx_idx": ctx_idx_list
    })
    return node_df
