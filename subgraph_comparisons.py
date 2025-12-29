from collections import namedtuple, Counter
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Set

import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from pandas import DataFrame
from attribution_graph_utils import create_or_load_graph, create_subgraph_from_selected_features, \
    get_subgraphs, get_feature, get_all_features
from common_utils import get_feature_from_node_id, create_node_id, Feature, get_output_token_from_clerp, \
    get_id_without_pos, get_top_output_logit_node, get_output_logits, get_linked_sources, get_links_from_node, \
    get_node_dict, get_frequencies, create_node_df
from constants import FEATURE_LAYER, FEATURE_ID

IntersectionMetrics = namedtuple("IntersectionMetrics",
                                 ['jaccard_index', 'relative_jaccard', 'frac_from_intersection',
                                  'shared_token', 'output_probability'])

CombinedMetrics = namedtuple("CombinedMetrics",
                             ['unique_frac', 'unique_weighted_frac', 'shared_frac', 'shared_weighted_frac',
                              'num_unique', 'num_shared', 'num_main'])

# Metrics for shared features across a set of similar prompts
# Core metrics:
# - num_shared: features shared by > 50% of prompts (primary metric)
# - num_prompts: number of prompts in the set
# - avg_features_per_prompt: average features per prompt
# Threshold counts (features shared by >= X% of prompts):
# - count_at_100pct: features in ALL prompts
# - count_at_75pct: features in >= 75% of prompts
# - count_at_50pct: features in >= 50% of prompts
# Per-prompt distribution (how many shared features each prompt contains):
# - shared_present_per_prompt: comma-separated string of counts for boxplot visualization
SharedFeatureMetrics = namedtuple("SharedFeatureMetrics",
                                  ['num_shared', 'num_prompts', 'avg_features_per_prompt',
                                   'count_at_100pct', 'count_at_75pct', 'count_at_50pct',
                                   'shared_present_per_prompt'])


def load_graphs_and_dfs(main_prompt: str, other_prompts: List[str], model: str, submodel: str,
                        graph_dir: Optional[str]) -> Tuple[dict, pd.DataFrame, List[dict], List[pd.DataFrame]]:
    """
    Loads all graphs and creates node dataframes once for reuse across methods.
    Returns (main_graph, main_df, other_graphs, other_dfs).
    """
    main_graph = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=main_prompt)
    main_df = create_node_df(main_graph, exclude_embeddings=True, exclude_errors=True, exclude_logits=True)
    main_df = main_df.drop_duplicates(subset=['layer', 'feature'])

    other_graphs = []
    other_dfs = []
    for prompt in other_prompts:
        graph = create_or_load_graph(graph_dir=graph_dir, model=model, submodel=submodel, prompt=prompt)
        df = create_node_df(graph, exclude_embeddings=True, exclude_errors=True, exclude_logits=True)
        df = df.drop_duplicates(subset=['layer', 'feature'])
        other_graphs.append(graph)
        other_dfs.append(df)

    return main_graph, main_df, other_graphs, other_dfs


def nodes_not_in(main_prompt: str, prompts2compare: List[str], model: str, submodel: str,
                 graph_dir: Optional[str], debug: bool = False,
                 return_metrics: bool = False,
                 preloaded: Optional[Tuple[dict, pd.DataFrame, List[dict], List[pd.DataFrame]]] = None
                 ) -> Tuple[dict, pd.DataFrame] | Tuple[dict, pd.DataFrame, dict]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are unique to the 'main_prompt'.
    Optionally accepts preloaded graphs/dfs to avoid reloading.
    """
    if preloaded:
        graph_metadata1, node_df1, other_graphs, other_dfs = preloaded
    else:
        graph_metadata1, node_df1, other_graphs, other_dfs = load_graphs_and_dfs(
            main_prompt, prompts2compare, model, submodel, graph_dir
        )

    unique_features = node_df1.copy()
    metrics = {}
    if debug:
        output_node = get_top_output_logit_node(graph_metadata1['nodes'])
        print(f"Main prompt output: ", output_node['clerp'])
    all_nodes = [{get_id_without_pos(node['node_id']): node for node in graph_metadata1['nodes']}]
    for i, (graph_metadata2, node_df2) in enumerate(zip(other_graphs, other_dfs)):
        metrics[prompts2compare[i]] = calculate_intersection_metrics(
            node_df1, node_df2, graph_metadata1, graph_metadata2, prompt_num=i + 1, debug=debug
        )
        unique_features = filter_for_unique_features(unique_features, node_df2)
        all_nodes.append({get_id_without_pos(node['node_id']): node for node in graph_metadata2['nodes']})

    if return_metrics:
        return graph_metadata1, unique_features, metrics
    return graph_metadata1, unique_features


def calculate_intersection_metrics(node_df1: pd.DataFrame, node_df2: pd.DataFrame,
                                   graph_metadata1: dict, graph_metadata2: dict, prompt_num: int = 2,
                                   debug: bool = False) -> IntersectionMetrics:
    """
    Calculates the jaccard index between two dataframes,
    the weighted jaccard index (sum(min(w1,w2)) / sum(max(w1,w2)) for all edges),
    and the ratio of intersecting features to all features for graph1.
    """
    node_df1 = node_df1.drop_duplicates(subset=['layer', 'feature'])
    node_df2 = node_df2.drop_duplicates(subset=['layer', 'feature'])
    intersection = pd.merge(node_df1, node_df2, on=['layer', 'feature'], how='inner')[['layer', 'feature']]
    links_lookup, _, output_node2 = get_links_overlap(graph_metadata1, graph_metadata2,
                                                      raise_if_no_matching_tokens=debug)
    if debug:
        print(f"Prompt {prompt_num} output: ", output_node2['clerp'])
        print(f"Graph URL: {graph_metadata2['metadata']['info']['url']}")

    # Weighted Jaccard: sum(min(w1, w2)) / sum(max(w1, w2)) for all edges
    min_weights = []
    max_weights = []
    for target, sources in links_lookup.items():
        for source, (w1, w2) in sources.items():
            min_weights.append(min(w1, w2))
            max_weights.append(max(w1, w2))
    relative_jaccard = sum(sorted(min_weights)) / sum(sorted(max_weights)) if max_weights else 0.0

    jaccard_index = len(intersection) / (len(node_df1) + len(node_df2) - len(intersection))
    frac_from_intersection = len(intersection) / len(node_df1)
    return IntersectionMetrics(jaccard_index, relative_jaccard, frac_from_intersection, output_node2['clerp'],
                               output_node2['token_prob'])


def get_links_overlap(graph_metadata1: dict,
                      graph_metadata2: dict,
                      raise_if_no_matching_tokens: bool = True) -> tuple[
    dict[str, Any], tuple[ndarray, ndarray], dict[str, Any]]:
    """
    Creates a nested dictionary with all target_ids at the top level, linked source_ids at the next level
    and finally link weights as the values which are represented as lists ([weight graph 1, weight graph 2]).
    If a edge does not appear in one of the graphs, its weight is set to 0. Additionally, the ratio of weights for
    intersecting edges to total for each graph is returned as a list.
    """
    output_logit_id1 = get_id_without_pos(get_top_output_logit_node(graph_metadata1['nodes'])['node_id'])
    output_node2 = [node for node in get_output_logits(graph_metadata2['nodes']) if
                    node['node_id'].startswith(output_logit_id1)]
    if not output_node2:
        if raise_if_no_matching_tokens:
            assert len(output_node2) == 1, "Output tokens don't match!"
        output_node2 = [get_top_output_logit_node(graph_metadata2['nodes'])]  # just grab top
    links1 = get_links_from_node(graph_metadata1, include_features_only=True)
    links2 = get_links_from_node(graph_metadata2,
                                 starting_node=output_node2[0] if len(output_node2) > 0 else None,
                                 include_features_only=True)
    links_lookup = {}

    # Collect weights in lists for stable summation
    total_weights_1 = []
    total_weights_2 = []
    intersection_weights_1 = []
    intersection_weights_2 = []

    def add_link(target, source, weight, index):
        target, source = get_id_without_pos(target), get_id_without_pos(source)
        if target not in links_lookup:
            links_lookup[target] = {}
        links_lookup[target][source] = [weight, 0] if index == 0 else [0, weight]

    for link in links1:
        add_link(link['target'], link['source'], link['weight'], 0)
        total_weights_1.append(link['weight'])

    for link in links2:
        target_id, source_id = get_id_without_pos(link['target']), get_id_without_pos(link['source'])
        if target_id in links_lookup and source_id in links_lookup[target_id]:
            intersection_weights_1.append(links_lookup[target_id][source_id][0])
            intersection_weights_2.append(link['weight'])
            links_lookup[target_id][source_id][-1] = link['weight']
        else:
            add_link(target_id, source_id, link['weight'], 1)
        total_weights_2.append(link['weight'])

    # Use sorted summation for numerical stability
    intersection = np.array([
        sum(sorted(intersection_weights_1)),
        sum(sorted(intersection_weights_2))
    ])
    total = np.array([
        sum(sorted(total_weights_1)),
        sum(sorted(total_weights_2))
    ])

    return links_lookup, (intersection, total), output_node2[0]


def get_relative_contribution(links: List[dict], all_features: pd.DataFrame, features_of_interest: pd.DataFrame,
                              output_node: dict):
    """
    Calculates how much the features of interest contribute overall to token prediction.
    """
    all_feature_ids = {create_node_id(feature, deliminator="_") for feature in all_features.itertuples()}
    all_feature_ids.add(get_id_without_pos(output_node["node_id"]))
    feature_of_interest_ids = {create_node_id(feature, deliminator="_") for feature in
                               features_of_interest.itertuples()}

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


def filter_features_for_subgraph(features_df: pd.DataFrame, graph_metadata: dict = None,
                                 model: str = None, submodel: str = None,
                                 filter_by_act_density: Optional[int] = None,
                                 filter_layers_less_than: int = None) -> pd.DataFrame:
    """
    Filters features by excluding embedding/output layers and optionally by activation density.
    """
    filter_layers = ['E']
    if graph_metadata:
        output_node = get_top_output_logit_node(graph_metadata['nodes'])
        filter_layers.append(output_node['layer'])
    if filter_layers_less_than is not None:
        filter_layers.extend([str(i) for i in range(filter_layers_less_than)])
    filtered = features_df[~features_df['layer'].isin(filter_layers)]

    if filter_by_act_density and len(filtered) > 0:
        feature_info = get_all_features(filtered, model=model, submodel=submodel)
        selected_features = [
            feature_row for info, feature_row in zip(feature_info, filtered.itertuples())
            if info['frac_nonzero'] * 100 < filter_by_act_density and info['frac_nonzero'] > 0
        ]
        debug_ = [
            info for info in feature_info
            if info['frac_nonzero'] * 100 < filter_by_act_density and info['frac_nonzero'] > 0
                    and int(info['layer'].split("-")[0]) > 1
        ]
        neg_str_ = {k:v for k, v in Counter([n_str for info in debug_ for n_str in info["neg_str"]]).items() if v > 1}
        pos_str_ =  {k:v for k, v in Counter([n_str for info in debug_ for n_str in info["pos_str"]]).items() if v > 1}
        filtered = pd.DataFrame(selected_features)
        if len(filtered) > 0:
            filtered = filtered[['layer', 'feature']] \
                if 'ctx_freq' not in filtered.columns else filtered[['layer', 'feature', 'ctx_freq']]

    return filtered


def get_most_freq_features_across_prompts(main_prompt: str, percent_shared: int,
                                          prompts2compare: List[str],
                                          model: str, submodel: str, graph_dir: Optional[str],
                                          preloaded: Optional[
                                              Tuple[dict, pd.DataFrame, List[dict], List[pd.DataFrame]]] = None
                                          ) -> pd.DataFrame:
    """
    Concatenates de-duplicated node dfs from all prompts and returns frequency counts.
    Each prompt's nodes are already de-duplicated, so ctx_freq represents how many prompts have each feature.
    """
    if preloaded:
        _, main_df, _, other_dfs = preloaded
    else:
        _, main_df, _, other_dfs = load_graphs_and_dfs(
            main_prompt, prompts2compare, model, submodel, graph_dir
        )

    # Combine all dfs (each already de-duplicated per prompt)
    all_dfs = [main_df] + other_dfs
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Get frequencies - counts how many prompts have each feature
    freq_df = get_frequencies(combined_df)
    threshold = (len(prompts2compare) + 1) * (percent_shared / 100)
    shared = freq_df[freq_df["ctx_freq"] >= threshold]
    return shared


def nodes_in(main_prompt: str, prompts2compare: List[str], model: str, submodel: str,
             graph_dir: Optional[str],
             preloaded: Optional[Tuple[dict, pd.DataFrame, List[dict], List[pd.DataFrame]]] = None
             ) -> Tuple[dict, pd.DataFrame]:
    """
    Compares the nodes across prompts and returns a dataframe of only the nodes which are shared across all prompts.
    Optionally accepts preloaded graphs/dfs to avoid reloading.
    """
    if preloaded:
        graph_metadata1, node_df1, _, other_dfs = preloaded
    else:
        graph_metadata1, node_df1, _, other_dfs = load_graphs_and_dfs(
            main_prompt, prompts2compare, model, submodel, graph_dir
        )

    overlapping_features = node_df1.copy()
    for i, node_df2 in enumerate(other_dfs):
        overlapping_features = pd.merge(overlapping_features, node_df2, on=['layer', 'feature'],
                                        how='inner', suffixes=(f'{i}a', f'{i}b'))
    # Return only unique layer/feature pairs
    overlapping_features = overlapping_features[['layer', 'feature']].drop_duplicates()
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
        target_feature = get_feature_from_node_id(link["target"], deliminator="_")
        if create_node_id(target_feature) in target_ids or link["target"] in target_ids:
            source_feature = get_feature_from_node_id(link["source"], deliminator="_")
            if create_node_id(source_feature) in source_ids or link["source"] in source_ids:
                selected_features.add((source_feature.layer, source_feature.feature))
                selected_features.add((target_feature.layer, target_feature.feature))
    return selected_features


def check_feature_presence(node_df: pd.DataFrame, layer: str, feature: str) -> bool:
    """Check if a specific feature exists in the node dataframe."""
    return len(node_df[(node_df['layer'] == layer) & (node_df['feature'] == feature)]) > 0


def calculate_shared_feature_metrics(main_prompt: str, sim_prompts: List[str],
                                      model: str, submodel: str, graph_dir: str,
                                      preloaded: Tuple[dict, pd.DataFrame, List[dict], List[pd.DataFrame]]
                                      ) -> SharedFeatureMetrics:
    """
    Calculate metrics about feature sharing across a set of prompts using
    get_most_freq_features_across_prompts at multiple thresholds with filtering.

    Args:
        main_prompt: The main prompt string
        sim_prompts: List of similar prompt strings to compare
        model: Model name
        submodel: Submodel name
        graph_dir: Directory containing graph data
        preloaded: Preloaded graphs and dataframes tuple

    Returns:
        SharedFeatureMetrics with core metrics, normalized metrics, and threshold counts.
    """
    _, main_df, _, other_dfs = preloaded
    all_dfs = [main_df] + other_dfs
    num_prompts = len(all_dfs)
    avg_features_per_prompt = sum(len(df.drop_duplicates(subset=['layer', 'feature'])) for df in all_dfs) / num_prompts

    def get_filtered_features_at_threshold(percent: int) -> pd.DataFrame:
        """Get features at threshold, applying same filtering as shared_features."""
        frequent_features = get_most_freq_features_across_prompts(
            main_prompt, percent, sim_prompts, model, submodel, graph_dir, preloaded)
        return filter_features_for_subgraph(
            frequent_features, model=model, submodel=submodel,
            filter_layers_less_than=1, filter_by_act_density=30)

    # Get filtered counts at different thresholds
    features_at_50pct = get_filtered_features_at_threshold(50)
    features_at_75pct = get_filtered_features_at_threshold(75)
    features_at_100pct = get_filtered_features_at_threshold(100)

    count_at_50pct = len(features_at_50pct)
    count_at_75pct = len(features_at_75pct)
    count_at_100pct = len(features_at_100pct)

    # num_shared uses 50% threshold as the primary metric
    num_shared = count_at_50pct

    # Calculate how many of the shared features (50% threshold) each prompt contains
    shared_present_per_prompt = []
    for node_df in all_dfs:
        count = sum(1 for row in features_at_50pct.itertuples()
                    if check_feature_presence(node_df, row.layer, row.feature))
        shared_present_per_prompt.append(count)

    # Store as comma-separated string for CSV compatibility
    shared_present_str = ','.join(str(c) for c in shared_present_per_prompt)

    return SharedFeatureMetrics(
        num_shared=num_shared,
        num_prompts=num_prompts,
        avg_features_per_prompt=avg_features_per_prompt,
        count_at_100pct=count_at_100pct,
        count_at_75pct=count_at_75pct,
        count_at_50pct=count_at_50pct,
        shared_present_per_prompt=shared_present_str
    )


def compare_prompt_subgraphs(main_prompt: str, diff_prompts: List[str], sim_prompts: List[str],
                             model: str, submodel: str, graph_dir: str, filter_by_act_density: int = None,
                             debug: bool = False,
                             feature_layer: str = FEATURE_LAYER,
                             feature_id: str = FEATURE_ID) -> Tuple[Optional[dict], Optional[dict], Optional[SharedFeatureMetrics]]:
    """
    Returns (metrics, feature_presence, shared_feature_metrics) where:
    - metrics: dict mapping prompt_id to IntersectionMetrics (from diff_prompts comparison)
    - feature_presence: dict mapping prompt_id to presence bool
    - shared_feature_metrics: SharedFeatureMetrics calculated from sim_prompts (or None if no sim_prompts)
    """
    overlapping_features: Optional[pd.DataFrame] = None
    shared_feature_metrics: Optional[SharedFeatureMetrics] = None
    unique_features: Optional[pd.DataFrame] = None
    graph_metadata: Optional[Dict[str, Any]] = None
    feature_presence: Optional[dict] = None

    # Finds nodes that are in the main prompt and not in any of the prompts in diff_prompts
    metrics = None
    if diff_prompts:
        # Load graphs and dfs once for reuse
        main_graph, main_df, other_graphs, other_dfs = load_graphs_and_dfs(
            main_prompt, diff_prompts, model, submodel, graph_dir
        )
        preloaded = (main_graph, main_df, other_graphs, other_dfs)

        graph_metadata, unique_features, metrics = nodes_not_in(main_prompt=main_prompt, prompts2compare=diff_prompts,
                                                                model=model, submodel=submodel, graph_dir=graph_dir,
                                                                debug=debug,
                                                                return_metrics=True,
                                                                preloaded=preloaded)
        feature_presence = {}
        for p, df, graph in zip([main_prompt] + diff_prompts, [main_df] + other_dfs, [main_graph] + other_graphs):
            feature_presence[p] = check_feature_presence(df, feature_layer, feature_id)

    # Finds nodes that are in the main prompt and also in all of the prompts in sim_prompts
    if sim_prompts:
        main_graph, main_df, other_graphs, other_dfs = load_graphs_and_dfs(
            main_prompt, sim_prompts, model, submodel, graph_dir
        )
        preloaded = (main_graph, main_df, other_graphs, other_dfs)
        graph_metadata, overlapping_features = nodes_in(main_prompt=main_prompt, prompts2compare=sim_prompts,
                                                        model=model, submodel=submodel, graph_dir=graph_dir,
                                                        preloaded=preloaded)
        frequent_features = get_most_freq_features_across_prompts(main_prompt=main_prompt, percent_shared=50,
                                                                  prompts2compare=sim_prompts,
                                                                  model=model, submodel=submodel, graph_dir=graph_dir,
                                                                  preloaded=preloaded)
        shared_features = filter_features_for_subgraph(
            frequent_features, model=model, submodel=submodel, filter_layers_less_than=1,
            filter_by_act_density=30,
        )

        # Calculate shared feature metrics across all prompts
        shared_feature_metrics = calculate_shared_feature_metrics(
            main_prompt, sim_prompts, model, submodel, graph_dir, preloaded)

        # Check feature presence in all graphs
        prompt2shared_features = {}
        for p, df, graph in zip([main_prompt] + sim_prompts, [main_df] + other_dfs, [main_graph] + other_graphs):
            prompt2shared_features[p] = [(row.layer, row.feature) for row in shared_features.itertuples()
                                         if check_feature_presence(df, row.layer, row.feature)]
            selected = pd.DataFrame({
                "feature": [f[-1] for f in prompt2shared_features[p]],
                "layer": [f[0] for f in prompt2shared_features[p]],
            })
            if len(selected) > 0:
                create_subgraph_from_selected_features(selected, graph,
                                                       list_name=f"Shared with others.")
                print("found:", graph['metadata']['info']['url'], get_top_output_logit_node(graph['nodes'])['clerp'])
            else:
                print("NOT found:", graph['metadata']['info']['url'],
                      get_top_output_logit_node(graph['nodes'])['clerp'])
    # Combine all features if both diff and sim prompts are provided. Otherwise, select relevant df.
    if diff_prompts and sim_prompts:
        features_of_interest = pd.merge(overlapping_features[['layer', 'feature']],
                                        unique_features[['layer', 'feature']], how='inner', on=['layer', 'feature'])
    else:
        features_of_interest = unique_features if unique_features is not None else overlapping_features

    assert features_of_interest is not None and graph_metadata is not None, \
        "Must provided either prompts to compare with or to contrast with."
    print(f"Neuronpedia Graph for Main Prompt: {graph_metadata['metadata']['info']['url']}")

    features_of_interest = filter_features_for_subgraph(
        features_of_interest, graph_metadata, model=model, submodel=submodel,
        filter_by_act_density=filter_by_act_density
    )
    timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    create_subgraph_from_selected_features(features_of_interest, graph_metadata,
                                           list_name=f"Features of Interest ({timestamp_str})")
    return metrics, feature_presence, shared_feature_metrics


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


def get_combined_links_lookup(graph_metadatas: List[dict]) -> dict[str, dict[str, list]]:
    """
    Creates a nested dictionary with all target_ids at the top level, linked source_ids at the next level
    and link weights as lists (one weight per graph, 0 if edge not present in that graph).
    """
    links_lookup = {}
    num_graphs = len(graph_metadatas)

    for graph_idx, graph_metadata in enumerate(graph_metadatas):
        links = get_links_from_node(graph_metadata, include_features_only=True)
        for link in links:
            target_id = get_id_without_pos(link['target'])
            source_id = get_id_without_pos(link['source'])
            if target_id not in links_lookup:
                links_lookup[target_id] = {}
            if source_id not in links_lookup[target_id]:
                links_lookup[target_id][source_id] = [0.0] * num_graphs
            links_lookup[target_id][source_id][graph_idx] = link['weight']

    return links_lookup


def calculate_edge_sharing_metrics(links_lookup: dict[str, dict[str, list]]) -> Tuple[float, float, float]:
    """
    Calculates weighted fractions for unique-to-main and shared-among-all edges.
    Returns (main_total_weight, unique_weight, shared_weight).
    """
    main_total_weight = 0.0
    unique_weight = 0.0
    shared_weight = 0.0

    for target, sources in links_lookup.items():
        for source, weights in sources.items():
            main_weight = weights[0]
            other_weights = weights[1:]

            if main_weight > 0:
                main_total_weight += main_weight

                # Edge is unique to main if no other graph has it
                if all(w == 0 for w in other_weights):
                    unique_weight += main_weight

                # Edge is shared if ALL graphs have it
                if all(w > 0 for w in other_weights):
                    shared_weight += main_weight

    return main_total_weight, unique_weight, shared_weight


def compare_prompts_combined(main_prompt: str, diff_prompts: List[str],
                             model: str, submodel: str, graph_dir: str,
                             debug: bool = False,
                             create_subgraphs: bool = True,
                             filter_by_act_density: Optional[int] = None) -> CombinedMetrics:
    """
    Compares main prompt against all diff prompts combined.
    Returns metrics for what's unique to main and what's shared among all.
    Loads graphs/dfs once and reuses across nodes_not_in and nodes_in.
    Optionally creates subgraphs for unique and shared features.
    """
    # Load all graphs and dataframes once
    preloaded = load_graphs_and_dfs(main_prompt, diff_prompts, model, submodel, graph_dir)
    main_graph, main_df, other_graphs, _ = preloaded

    if debug:
        output_node = get_top_output_logit_node(main_graph['nodes'])
        print(f"Main prompt output: {output_node['clerp']}")
    print(f"Graph URL: {main_graph['metadata']['info']['url']}")

    # Get unique features using existing method with preloaded data
    _, unique_features = nodes_not_in(
        main_prompt=main_prompt, prompts2compare=diff_prompts,
        model=model, submodel=submodel, graph_dir=graph_dir, debug=False,
        preloaded=preloaded
    )

    # Get shared features using existing method with preloaded data
    _, shared_features = nodes_in(
        main_prompt=main_prompt, prompts2compare=diff_prompts,
        model=model, submodel=submodel, graph_dir=graph_dir,
        preloaded=preloaded
    )

    # Calculate feature-based fractions
    num_main = len(main_df)
    num_unique = len(unique_features)
    num_shared = len(shared_features)

    unique_frac = num_unique / num_main if num_main > 0 else 0.0
    shared_frac = num_shared / num_main if num_main > 0 else 0.0

    # Calculate weighted metrics using edge weights
    all_graphs = [main_graph] + other_graphs
    links_lookup = get_combined_links_lookup(all_graphs)
    main_total_weight, unique_weight, shared_weight = calculate_edge_sharing_metrics(links_lookup)

    unique_weighted_frac = unique_weight / main_total_weight if main_total_weight > 0 else 0.0
    shared_weighted_frac = shared_weight / main_total_weight if main_total_weight > 0 else 0.0

    if debug:
        print(f"Main features: {num_main}, Unique: {num_unique} ({unique_frac:.3f}), "
              f"Shared: {num_shared} ({shared_frac:.3f})")
        print(f"Weighted - Unique: {unique_weighted_frac:.3f}, Shared: {shared_weighted_frac:.3f}")

    # Create subgraphs for unique and shared features
    if create_subgraphs:
        timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")

        # Filter and create subgraph for unique features
        unique_filtered = filter_features_for_subgraph(unique_features, main_graph)
        if len(unique_filtered) > 0:
            create_subgraph_from_selected_features(
                unique_filtered, main_graph,
                list_name=f"Unique Features ({timestamp_str})"
            )

        # Filter and create subgraph for shared features
        shared_filtered = filter_features_for_subgraph(
            shared_features, main_graph, model=model, submodel=submodel,
            filter_by_act_density=filter_by_act_density
        )
        if len(shared_filtered) > 0:
            create_subgraph_from_selected_features(
                shared_filtered, main_graph,
                list_name=f"Shared Features ({timestamp_str})"
            )

        # Create subgraph for shared features with frequency > 1
        main_node_df_full = create_node_df(main_graph, exclude_embeddings=True, exclude_errors=True,
                                           exclude_logits=True)
        freq_df = get_frequencies(main_node_df_full)
        frequent_features = freq_df[freq_df['ctx_freq'] > 1][['layer', 'feature']]
        shared_frequent = pd.merge(shared_filtered, frequent_features, on=['layer', 'feature'], how='inner')
        if len(shared_frequent) > 0:
            create_subgraph_from_selected_features(
                shared_frequent, main_graph,
                list_name=f"Shared Features Frequent Fliers ({timestamp_str})"
            )

    return CombinedMetrics(unique_frac, unique_weighted_frac, shared_frac, shared_weighted_frac,
                           num_unique, num_shared, num_main)
