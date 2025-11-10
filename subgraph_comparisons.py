from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Set

import pandas as pd
from pandas import DataFrame

from attribution_graph_utils import create_node_df, create_or_load_graph, create_subgraph_from_selected_features, \
    get_subgraphs, get_linked_sources
from common_utils import get_feature_from_node_id, create_node_id, Feature, get_output_token_from_clerp


def nodes_not_in(main_prompt: str, prompts2compare: List[str], model: str, submodel: str,
                 graph_dir: Optional[str]) -> Tuple[dict, pd.DataFrame]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are unique to the 'main_prompt'
    """
    graph_metadata1 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=main_prompt)
    node_df1 = create_node_df(graph_metadata1)
    unique_features = node_df1
    for prompt in prompts2compare:
        graph_metadata2 = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel,
                                               prompt=prompt)

        node_df2 = create_node_df(graph_metadata2)
        unique_features = filter_for_unique_features(unique_features, node_df2)
    return graph_metadata1, unique_features


def filter_for_unique_features(node_df1: DataFrame, node_df2: DataFrame) -> DataFrame:
    """
    Filters out features that are in node_df2 and keeps only features unique to node_df1.
    """
    merged_df = node_df1.merge(node_df2, on=['layer', 'feature'],
                               how='left', indicator=True).drop_duplicates()
    node_df1 = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    return node_df1


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
                             model: str, submodel: str, graph_dir: str):
    overlapping_features: Optional[pd.DataFrame] = None
    unique_features: Optional[pd.DataFrame] = None
    graph_metadata: Optional[Dict[str, Any]] = None

    # Finds nodes that are in the main prompt and not in any of the prompts in diff_prompts
    if diff_prompts:
        graph_metadata, unique_features = nodes_not_in(main_prompt=main_prompt, prompts2compare=diff_prompts,
                                                       model=model, submodel=submodel, graph_dir=graph_dir)

    # Finds nodes that are in the main prompt and also in all of the prompts in sim_prompts
    if sim_prompts:
        graph_metadata, overlapping_features = nodes_in(main_prompt=main_prompt, prompts2compare=sim_prompts,
                                                        model=model, submodel=submodel, graph_dir=graph_dir)

    # Combine all features if both diff and sim prompts are provided. Otherwise, select relevant df.
    if diff_prompts and sim_prompts:
        features_of_interest = pd.merge(overlapping_features, unique_features, how='inner', on=['layer', 'feature'])
    else:
        features_of_interest = unique_features if unique_features is not None else overlapping_features

    assert features_of_interest is not None and graph_metadata is not None, \
        "Must provided either prompts to compare with or to contrast with."
    print(f"Neuronpedia Graph for Main Prompt: {graph_metadata['metadata']['info']['url']}")

    output_node = graph_metadata["nodes"][-1]
    features_of_interest = features_of_interest[~features_of_interest['layer'].isin(['0', 'E', output_node['layer']])]

    timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    create_subgraph_from_selected_features(features_of_interest, graph_metadata,
                                           list_name=f"Features of Interest ({timestamp_str})")

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

    create_subgraph_from_selected_features(node_df, graph_metadata, f"unique features for {token_of_interest}",
                                                 include_output_node=False)


