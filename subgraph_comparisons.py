from collections import namedtuple
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Set

import numpy as np
import pandas as pd
from pandas import DataFrame

from attribution_graph_utils import create_node_df, create_or_load_graph, create_subgraph_from_selected_features, \
    get_subgraphs, get_linked_sources, get_top_output_logit_node, merge_node_dfs, get_links_from_node, \
    create_subgraph_from_links, get_output_logits
from common_utils import get_feature_from_node_id, create_node_id, Feature, get_output_token_from_clerp, get_id_without_pos

IntersectionMetrics = namedtuple("IntersectionMetrics", ['jaccard_index', 'relative_contribution', 'frac_from_intersection'])

def nodes_not_in(main_prompt: str, prompts2compare: List[str], model: str, submodel: str,
                 graph_dir: Optional[str], debug: bool = False,
                 return_metrics: bool = False) -> Tuple[dict, pd.DataFrame] | Tuple[dict, pd.DataFrame, dict]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are unique to the 'main_prompt'
    """
    graph_metadata1 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=main_prompt)
    node_df1 = create_node_df(graph_metadata1, exclude_embeddings=True, exclude_errors=True,
                              exclude_logits=True)
    node_df1 = node_df1.drop_duplicates(subset=['layer', 'feature'])
    unique_features = node_df1
    metrics = {}
    if debug:
        output_node = get_top_output_logit_node(graph_metadata1['nodes'])
        print(f"Main prompt output: ", output_node['clerp'])
    for i, prompt in enumerate(prompts2compare):
        graph_metadata2 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                               prompt=prompt)

        node_df2 = create_node_df(graph_metadata2,  exclude_embeddings=True, exclude_errors=True,
                                  exclude_logits=True)
        node_df2 = node_df2.drop_duplicates(subset=['layer', 'feature'])
        if debug:
            output_node = get_top_output_logit_node(graph_metadata2['nodes'])
            print(f"Prompt {i+1} output: ", output_node['clerp'])
        metrics[prompt] = calculate_intersection_metrics(node_df1, node_df2, graph_metadata1, graph_metadata2,
                                                         debug=debug)
        unique_features = filter_for_unique_features(unique_features, node_df2)
    if return_metrics:
        return graph_metadata1, unique_features, metrics
    return graph_metadata1, unique_features

def run_error_analysis(prompts: List[str], model: str, submodel: str,
                 graph_dir: Optional[str]) -> dict[str, float]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are unique to the 'main_prompt'
    """
    metrics = {}
    for prompt in prompts:
        graph_metadata = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                               prompt=prompt)
        metrics[prompt] = calculate_error_contributions(graph_metadata)
    return metrics

def calculate_intersection_metrics(node_df1: pd.DataFrame, node_df2: pd.DataFrame,
                                   graph_metadata1: dict,  graph_metadata2: dict,
                                   debug: bool = False) -> IntersectionMetrics:
    """
    Calculates the jaccard index between two dataframes.
    """
    node_df1 = node_df1.drop_duplicates(subset=['layer', 'feature'])
    node_df2 = node_df2.drop_duplicates(subset=['layer', 'feature'])
    intersection = pd.merge(node_df1, node_df2, on=['layer', 'feature'], how='inner')[['layer', 'feature']]
    _, relative_contributions = get_links_union(graph_metadata1, graph_metadata2)
    # relative_contribution = get_relative_contribution(links1, node_df1, intersection,
    #                                                   output_node=get_top_output_logit_node(graph_metadata1['nodes']))
    jaccard_index = len(intersection) / (len(node_df1) + len(node_df2) - len(intersection))
    frac_from_intersection = len(intersection) / len(node_df1)
    return IntersectionMetrics(jaccard_index, relative_contributions[1], frac_from_intersection)

def get_links_union(graph_metadata1: dict, graph_metadata2: dict) -> tuple[dict, list]:
    """
    Creates a dictionary with all links, combining weights as tuple (weight1, weight2).
    """
    output_logit_id1 = get_id_without_pos(get_top_output_logit_node(graph_metadata1['nodes'])['node_id'])
    output_node2 = [node for node in get_output_logits(graph_metadata2['nodes']) if node['node_id'].startswith(output_logit_id1)]
    links1 = get_links_from_node(graph_metadata1, include_features_only=True)
    links2 = get_links_from_node(graph_metadata2,
                                 starting_node=output_node2[0] if len(output_node2) > 0 else None,
                                 include_features_only=True)
    links_lookup = {}
    intersection = np.array([0.0, 0.0])
    total =  np.array([0.0, 0.0])
    def add_link(target, source, weight, index):
        target, source = get_id_without_pos(target), get_id_without_pos(source)
        if target not in links_lookup:
            links_lookup[target] = {}
        links_lookup[target][source] = [weight, 0] if index == 0 else [0, weight]
        total[index] += weight
    [add_link(link['target'], link['source'], link['weight'], 0) for link in links1]
    for link in links2:
        target_id, source_id = get_id_without_pos(link['target']), get_id_without_pos(link['source'])
        if target_id in links_lookup and source_id in links_lookup[target_id]:
            links_lookup[target_id][source_id][-1] = link['weight']
            intersection += np.array(links_lookup[target_id][source_id])
            total[-1] +=  link['weight']
        else:
            add_link(target_id, source_id, link['weight'], 1)
    return links_lookup, (intersection / total).tolist()

def calculate_error_contributions(graph_metadata: dict, hops: int = 1) -> float:
    """
    Calculates the ratio of error contributions to overall.
    """
    node_dict = {node["node_id"]: node for node in graph_metadata['nodes']}
    output_node = get_top_output_logit_node(graph_metadata['nodes'])
    links = get_links_from_node(graph_metadata, output_node, hops=hops)
    error_contribution = 0
    total_contribution = 0
    for link in links:
        source_id = link["source"]
        if node_dict[source_id].get("feature_type") == "mlp reconstruction error":
            error_contribution += 1
        total_contribution += 1
    return error_contribution / total_contribution

def get_relative_contribution(links: List[dict], all_features: pd.DataFrame, features_of_interest: pd.DataFrame,
                              output_node: dict):
    """
    Calculates how much the features of interest contribute overall to token prediction.
    """
    all_feature_ids = {create_node_id(feature, deliminator="_") for feature in all_features.itertuples()}
    all_feature_ids.add(get_id_without_pos(output_node["node_id"]))
    feature_of_interest_ids = {create_node_id(feature, deliminator="_") for feature in features_of_interest.itertuples()}

    all_contribution = 0
    subset_contribution = 0

    for link in links:
        target_id = get_id_without_pos(link["target"])
        source_id = get_id_without_pos(link["source"])
        if not (target_id in all_feature_ids or source_id in all_feature_ids):
            debug = True
            continue
        all_contribution += link['weight']
        if target_id in feature_of_interest_ids or source_id in feature_of_interest_ids:
            subset_contribution += link['weight']
    return subset_contribution / all_contribution

def filter_for_unique_features(node_df1: DataFrame, node_df2: DataFrame) -> DataFrame:
    """
    Filters out features that are in node_df2 and keeps only features unique to node_df1.
    """
    diff = (
        node_df1
        .merge(node_df2[['layer', 'feature']].drop_duplicates(), on=['layer', 'feature'], how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    return diff


def nodes_in(main_prompt: str, prompts2compare: List[str], model: str, submodel: str,
             graph_dir: Optional[str]) -> Tuple[dict, pd.DataFrame]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are unique to the 'main_prompt'
    """
    graph_metadata1 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=main_prompt)
    node_df1 = create_node_df(graph_metadata1)
    overlapping_features = node_df1
    for i, prompt in enumerate(prompts2compare):
        graph_metadata2 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                               prompt=prompt)

        node_df2 = create_node_df(graph_metadata2)
        overlapping_features = pd.merge(overlapping_features, node_df2, on=['layer', 'feature'],
                                        how='inner', suffixes=(f'{i}a', f'{i}b'))
    return graph_metadata1, overlapping_features

def select_features_by_links(graph_metadata: dict, target_ids: str | Set[str],
                          source_ids: str | Set[str], pos_links_only: bool = True) -> Set[Feature]:
    """
    Selects features based on whether they are connected to the given target ids or source ids.
    """
    target_ids = {target_ids} if not isinstance(target_ids, set) else target_ids
    source_ids = {source_ids} if not isinstance(source_ids, set) else source_ids

    selected_features = set()
    for link in graph_metadata["links"]:
        if pos_links_only and link["weight"] < 0:
            continue
        target_feature =  get_feature_from_node_id(link["target"], deliminator="_")
        if create_node_id(target_feature) in target_ids or link["target"] in target_ids:
            source_feature =  get_feature_from_node_id(link["source"], deliminator="_")
            if create_node_id(source_feature) in source_ids or link["source"] in source_ids:
                selected_features.add((source_feature.layer, source_feature.feature))
                selected_features.add((target_feature.layer, target_feature.feature))
    return selected_features


def compare_prompt_subgraphs(main_prompt: str, diff_prompts: List[str], sim_prompts: List[str],
                             model: str, submodel: str, graph_dir: str, debug: bool = False) -> Optional[dict]:
    overlapping_features: Optional[pd.DataFrame] = None
    unique_features: Optional[pd.DataFrame] = None
    graph_metadata: Optional[Dict[str, Any]] = None

    # Finds nodes that are in the main prompt and not in any of the prompts in diff_prompts
    metrics = None
    if diff_prompts:
        graph_metadata, unique_features, metrics = nodes_not_in(main_prompt=main_prompt, prompts2compare=diff_prompts,
                                                       model=model, submodel=submodel, graph_dir=graph_dir, debug=debug,
                                                       return_metrics=True)

    # Finds nodes that are in the main prompt and also in all of the prompts in sim_prompts
    if sim_prompts:
        graph_metadata, overlapping_features = nodes_in(main_prompt=main_prompt, prompts2compare=sim_prompts,
                                                        model=model, submodel=submodel, graph_dir=graph_dir)

    # Combine all features if both diff and sim prompts are provided. Otherwise, select relevant df.
    if diff_prompts and sim_prompts:
        features_of_interest = pd.merge(overlapping_features[['layer', 'feature']], unique_features[['layer', 'feature']], how='inner', on=['layer', 'feature'])
    else:
        features_of_interest = unique_features if unique_features is not None else overlapping_features

    assert features_of_interest is not None and graph_metadata is not None, \
        "Must provided either prompts to compare with or to contrast with."
    print(f"Neuronpedia Graph for Main Prompt: {graph_metadata['metadata']['info']['url']}")

    output_node = get_top_output_logit_node(graph_metadata["nodes"])
    features_of_interest = features_of_interest[~features_of_interest['layer'].isin(['0', 'E', output_node['layer']])]

    timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    create_subgraph_from_selected_features(features_of_interest, graph_metadata,
                                           list_name=f"Features of Interest ({timestamp_str})")
    return metrics

def unique_features_for_token(token_of_interest: str,
                              output_token_to_linked_features: dict[str, Set[str]]) -> list[Feature]:
    # all features except those linked to token of interest
    other_features = set()
    for token, node_ids in output_token_to_linked_features.items():
        if token != token_of_interest:
            other_features.update(node_ids)
    assert token_of_interest in output_token_to_linked_features, f"Token {token_of_interest} not found"
    unique_node_ids = output_token_to_linked_features[token_of_interest].difference(other_features)
    unique_features = [get_feature_from_node_id(node_id, deliminator="_") for node_id in unique_node_ids]
    return unique_features

def compare_token_subgraphs(main_prompt: str, token_of_interest: str, model: str, submodel: str,
                            graph_dir: str, tok_k_outputs: int = 5):
    graph_metadata = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=main_prompt)
    get_subgraphs(graph_metadata)
    print(graph_metadata["metadata"]["info"]["url"])

    output_nodes = [node for node in graph_metadata["nodes"] if node["feature_type"] == "logit"]
    top_nodes = output_nodes[:tok_k_outputs]

    output_token_to_id = {get_output_token_from_clerp(node): node["node_id"] for node in top_nodes}
    output_token_to_linked_features = {token: {node_id} for token, node_id in output_token_to_id.items()}

    ## Update output_token_to_linked_features with a set of linked features for each output token
    newly_added = True
    while newly_added:
        newly_added = get_linked_sources(graph_metadata, output_token_to_linked_features, positive_only=True)

    unique_features = unique_features_for_token(token_of_interest, output_token_to_linked_features)

    # Create df for features
    node_dict = [{"layer": feature.layer, "feature": feature.feature} for feature in unique_features if
                 int(feature.layer) > 1]  # list of all unique features at layers higher than 1
    node_df = pd.DataFrame(node_dict)

    create_subgraph_from_selected_features(node_df, graph_metadata, f"Unique Features for {token_of_interest}",
                                                 include_output_node=False)


