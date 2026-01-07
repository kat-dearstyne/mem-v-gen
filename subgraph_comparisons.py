from collections import namedtuple, Counter
from datetime import datetime
from typing import List, Optional, Tuple, Any, Set, Dict

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from src.constants import FEATURE_LAYER, FEATURE_ID
from src.graph_analyzer import (GraphAnalyzer, ComparisonMetrics,
                                 SharedFeatureMetrics as SharedFeatureMetricsEnum,
                                 SubgraphComparisonMetrics)
from src.graph_manager import GraphManager, Feature
from src.neuronpedia_manager import NeuronpediaManager, GraphConfig

CombinedMetrics = namedtuple("CombinedMetrics",
                             ['unique_frac', 'unique_weighted_frac', 'shared_frac', 'shared_weighted_frac',
                              'num_unique', 'num_shared', 'num_main'])

def compare_prompt_subgraphs(main_prompt: str, diff_prompts: List[str], sim_prompts: List[str],
                             model: str, submodel: str, graph_dir: str, filter_by_act_density: int = None,
                             debug: bool = False,
                             feature_layer: str = FEATURE_LAYER,
                             feature_id: str = FEATURE_ID,
                             metrics2run: Optional[Set[ComparisonMetrics]] = None,
                             prompt2ids: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare subgraphs across prompts and return metrics for diff and sim groups separately.

    Args:
        main_prompt: The main prompt to analyze.
        diff_prompts: Prompts to compare against (finds unique features).
        sim_prompts: Similar prompts to compare (finds shared features).
        model: Model name.
        submodel: Submodel name.
        graph_dir: Directory containing graph data.
        filter_by_act_density: Optional activation density filter threshold.
        debug: Whether to print debug output.
        feature_layer: Layer of feature to check presence for.
        feature_id: ID of feature to check presence for.
        metrics2run: Set of ComparisonMetrics to calculate on both diff and sim groups.
            If None, runs all comparison metrics.
        prompt2ids: Optional mapping from prompt strings to their display IDs.
            Results will be keyed by these IDs if provided.

    Returns:
        Tuple of (diff_results, sim_results), each a dictionary with keys from SubgraphComparisonMetrics:
        - 'intersection_metrics': dict mapping prompt_id to comparison metrics
        - 'feature_presence': dict mapping prompt_id to presence bool
        - 'shared_features': dict of shared feature metrics
    """
    # Default to all comparison metrics if not specified
    if metrics2run is None:
        metrics2run = {m for m in ComparisonMetrics}

    # Build prompt to ID mapping - use provided or create identity mapping
    if prompt2ids is None:
        all_prompts = [main_prompt] + diff_prompts + sim_prompts
        prompt2ids = {p: p for p in all_prompts}

    config = GraphConfig(model=model, submodel=submodel)
    neuronpedia_manager = NeuronpediaManager(graph_dir=graph_dir, config=config)

    # Separate results dicts for diff and sim
    diff_results: Dict[str, Any] = {}
    sim_results: Dict[str, Any] = {}

    overlapping_features: Optional[pd.DataFrame] = None
    unique_features: Optional[pd.DataFrame] = None

    # Build prompt ID mappings using prompt2ids
    main_prompt_id = prompt2ids[main_prompt]
    diff_prompt_ids = [prompt2ids[p] for p in diff_prompts] if diff_prompts else []
    sim_prompt_ids = [prompt2ids[p] for p in sim_prompts] if sim_prompts else []

    # Create single analyzer with all prompts
    all_prompts_dict = {main_prompt_id: main_prompt}
    all_prompts_dict.update({p_id: p for p_id, p in zip(diff_prompt_ids, diff_prompts)})
    all_prompts_dict.update({p_id: p for p_id, p in zip(sim_prompt_ids, sim_prompts)})
    analyzer = GraphAnalyzer(prompts=all_prompts_dict, neuronpedia_manager=neuronpedia_manager)
    main_graph, _ = analyzer.get_graph_and_df(main_prompt_id)

    # Finds nodes that are in the main prompt and not in any of the prompts in diff_prompts
    if diff_prompts:
        unique_features, intersection_metrics = analyzer.nodes_not_in(
            main_prompt_id=main_prompt_id,
            comparison_prompts=diff_prompt_ids,
            verbose=debug,
            metrics2run=metrics2run
        )
        if intersection_metrics:
            diff_results[SubgraphComparisonMetrics.INTERSECTION_METRICS.value] = intersection_metrics

        feature_presence = {}
        for p_id in [main_prompt_id] + diff_prompt_ids:
            graph, _ = analyzer.get_graph_and_df(p_id)
            feature_presence[p_id] = graph.check_feature_presence(feature_layer, feature_id)
        diff_results[SubgraphComparisonMetrics.FEATURE_PRESENCE.value] = feature_presence

    # Finds nodes that are in the main prompt and also in all of the prompts in sim_prompts
    if sim_prompts:
        overlapping_features, sim_intersection_metrics = analyzer.nodes_in(
            main_prompt_id=main_prompt_id,
            comparison_prompts=sim_prompt_ids,
            verbose=debug,
            metrics2run=metrics2run
        )
        if sim_intersection_metrics:
            sim_results[SubgraphComparisonMetrics.INTERSECTION_METRICS.value] = sim_intersection_metrics

        shared_features = analyzer.get_most_freq_features_across_prompts(
            percent_shared=50,
            prompts2compare=sim_prompt_ids
        )

        # Calculate shared feature metrics across all prompts
        shared_feature_metrics = analyzer.calculate_shared_feature_metrics(
            comparison_prompts=[main_prompt_id] + sim_prompt_ids
        )
        if shared_feature_metrics:
            sim_results[SubgraphComparisonMetrics.SHARED_FEATURES.value] = shared_feature_metrics

        # Check feature presence in all graphs and create subgraphs
        feature_presence = {}
        prompt2shared_features = {}
        for p_id in [main_prompt_id] + sim_prompt_ids:
            graph, _ = analyzer.get_graph_and_df(p_id)
            feature_presence[p_id] = graph.check_feature_presence(feature_layer, feature_id)
            prompt2shared_features[p_id] = [(row.layer, row.feature) for row in shared_features.itertuples()
                                            if graph.check_feature_presence(row.layer, row.feature)]
            selected = pd.DataFrame({
                "feature": [f[-1] for f in prompt2shared_features[p_id]],
                "layer": [f[0] for f in prompt2shared_features[p_id]],
            })
            if len(selected) > 0:
                neuronpedia_manager.create_subgraph_from_selected_features(selected, graph,
                                                                           list_name=f"Shared with others.")
                print("found:", graph.url, graph.get_top_output_logit_node()['clerp'])
            else:
                print("NOT found:", graph.url, graph.get_top_output_logit_node()['clerp'])
        sim_results[SubgraphComparisonMetrics.FEATURE_PRESENCE.value] = feature_presence

    # Combine all features if both diff and sim prompts are provided. Otherwise, select relevant df.
    if diff_prompts and sim_prompts:
        features_of_interest = pd.merge(overlapping_features[GraphManager.NODE_COLUMNS],
                                        unique_features[GraphManager.NODE_COLUMNS],
                                        how='inner', on=GraphManager.NODE_COLUMNS)
    else:
        features_of_interest = unique_features if unique_features is not None else overlapping_features

    assert features_of_interest is not None and main_graph is not None, \
        "Must provide either prompts to compare with or to contrast with."
    print(f"Neuronpedia Graph for Main Prompt: {main_graph.url}")

    features_of_interest = neuronpedia_manager.filter_features_for_subgraph(
        features_of_interest, main_graph,
        filter_by_act_density=filter_by_act_density
    )
    timestamp_str = datetime.now().strftime("%m-%d-%y %H:%M:%S")
    neuronpedia_manager.create_subgraph_from_selected_features(features_of_interest, main_graph,
                                                               list_name=f"Features of Interest ({timestamp_str})")
    return diff_results, sim_results


def unique_features_for_token(token_of_interest: str,
                              output_token_to_linked_features: dict[str, Set[str]]) -> list[Feature]:
    """
    Gets features unique to a specific token (not linked to any other tokens).
    """
    # all features except those linked to token of interest
    other_features = set()
    for token, node_ids in output_token_to_linked_features.items():
        if token != token_of_interest:
            other_features.update(node_ids)
    assert token_of_interest in output_token_to_linked_features, f"Token {token_of_interest} not found"
    unique_node_ids = output_token_to_linked_features[token_of_interest].difference(other_features)
    unique_features = [GraphManager.get_feature_from_node_id(node_id, deliminator="_") for node_id in unique_node_ids]
    return unique_features


def compare_token_subgraphs(main_prompt: str, token_of_interest: str, model: str, submodel: str,
                            graph_dir: str, tok_k_outputs: int = 5):
    """
    Compares subgraphs for different output tokens and creates a subgraph of unique features.
    """
    config = GraphConfig(model=model, submodel=submodel)
    neuronpedia_manager = NeuronpediaManager(graph_dir=graph_dir, config=config)

    graph = neuronpedia_manager.create_or_load_graph(prompt=main_prompt)
    neuronpedia_manager.get_subgraphs(graph)
    print(graph.url)

    output_nodes = [node for node in graph.nodes if node["feature_type"] == "logit"]
    top_nodes = output_nodes[:tok_k_outputs]

    output_token_to_id = {GraphManager.get_output_token_from_clerp(node): node["node_id"] for node in top_nodes}
    output_token_to_linked_features = {token: {node_id} for token, node_id in output_token_to_id.items()}

    ## Update output_token_to_linked_features with a set of linked features for each output token
    newly_added = True
    while newly_added:
        newly_added = graph.get_linked_sources(output_token_to_linked_features, positive_only=True)

    unique_features = unique_features_for_token(token_of_interest, output_token_to_linked_features)

    # Create df for features
    node_dict = [{"layer": feature.layer, "feature": feature.feature} for feature in unique_features if
                 int(feature.layer) > 1]  # list of all unique features at layers higher than 1
    node_df = pd.DataFrame(node_dict)

    neuronpedia_manager.create_subgraph_from_selected_features(node_df, graph,
                                                         f"Unique Features for {token_of_interest}",
                                                         include_output_node=False)


def get_combined_links_lookup(graphs: List[GraphManager]) -> dict[str, dict[str, list]]:
    """
    Creates a nested dictionary with all target_ids at the top level, linked source_ids at the next level
    and link weights as lists (one weight per graph, 0 if edge not present in that graph).
    """
    links_lookup = {}
    num_graphs = len(graphs)

    for graph_idx, graph in enumerate(graphs):
        links = graph.get_links_from_node(include_features_only=True)
        for link in links:
            target_id = GraphManager.get_id_without_pos(link['target'])
            source_id = GraphManager.get_id_without_pos(link['source'])
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
    Optionally creates subgraphs for unique and shared features.
    """
    config = GraphConfig(model=model, submodel=submodel)
    neuronpedia_manager = NeuronpediaManager(graph_dir=graph_dir, config=config)

    # Build prompt ID mappings
    main_prompt_id = "main"
    diff_prompt_ids = [f"diff_{i}" for i in range(len(diff_prompts))]

    prompts_dict = {main_prompt_id: main_prompt}
    prompts_dict.update({p_id: p for p_id, p in zip(diff_prompt_ids, diff_prompts)})

    analyzer = GraphAnalyzer(prompts=prompts_dict, neuronpedia_manager=neuronpedia_manager)
    main_graph, main_df = analyzer.get_graph_and_df(main_prompt_id)

    if debug:
        output_node = main_graph.get_top_output_logit_node()
        print(f"Main prompt output: {output_node['clerp']}")
    print(f"Graph URL: {main_graph.url}")

    # Get unique features
    unique_features = analyzer.nodes_not_in(
        main_prompt_id=main_prompt_id,
        comparison_prompts=diff_prompt_ids,
        verbose=False
    )

    # Get shared features
    shared_features = analyzer.nodes_in(
        main_prompt_id=main_prompt_id,
        comparison_prompts=diff_prompt_ids,
        verbose=False
    )

    # Calculate feature-based fractions
    num_main = len(main_df)
    num_unique = len(unique_features)
    num_shared = len(shared_features)

    unique_frac = num_unique / num_main if num_main > 0 else 0.0
    shared_frac = num_shared / num_main if num_main > 0 else 0.0

    # Calculate weighted metrics using edge weights
    other_graphs = [analyzer.graphs[p_id] for p_id in diff_prompt_ids]
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
        unique_filtered = neuronpedia_manager.filter_features_for_subgraph(unique_features, main_graph)
        if len(unique_filtered) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                unique_filtered, main_graph,
                list_name=f"Unique Features ({timestamp_str})"
            )

        # Filter and create subgraph for shared features
        shared_filtered = neuronpedia_manager.filter_features_for_subgraph(
            shared_features, main_graph,
            filter_by_act_density=filter_by_act_density
        )
        if len(shared_filtered) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                shared_filtered, main_graph,
                list_name=f"Shared Features ({timestamp_str})"
            )

        # Create subgraph for shared features with frequency > 1
        main_node_df_full = main_graph.create_node_df(exclude_embeddings=True, exclude_errors=True,
                                                      exclude_logits=True)
        freq_df = main_graph.get_frequencies_from_graph(main_node_df_full)
        frequent_features = freq_df[freq_df['ctx_freq'] > 1][GraphManager.NODE_COLUMNS]
        shared_frequent = pd.merge(shared_filtered, frequent_features, on=GraphManager.NODE_COLUMNS, how='inner')
        if len(shared_frequent) > 0:
            neuronpedia_manager.create_subgraph_from_selected_features(
                shared_frequent, main_graph,
                list_name=f"Shared Features Frequent Fliers ({timestamp_str})"
            )

    return CombinedMetrics(unique_frac, unique_weighted_frac, shared_frac, shared_weighted_frac,
                           num_unique, num_shared, num_main)
