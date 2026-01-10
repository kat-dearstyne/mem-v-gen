import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

Feature = namedtuple("Feature", ("layer", "feature"))

class GraphManager:
    """Manages graph metadata and provides methods for graph operations."""
    NODE_COLUMNS = ['layer', 'feature']

    def __init__(self, graph_metadata: dict):
        """
        Initialize GraphManager with graph metadata.

        Args:
            graph_metadata: Dictionary containing nodes, links, and metadata.
        """
        self.graph_metadata = graph_metadata
        self.nodes = graph_metadata["nodes"]
        self.links = graph_metadata["links"]
        self.metadata = graph_metadata["metadata"]
        self.prompt = self.metadata["prompt"]
        self.url = self.metadata["info"]["url"]
        self.node_dict = self.get_node_dict()

    def create_node_df(self, include_errors_by_pos: bool = False,
                       exclude_embeddings: bool = False, exclude_errors: bool = False,
                       exclude_logits: bool = False, drop_duplicates: bool = False) -> pd.DataFrame:
        """
        Extract node information from graph metadata into a DataFrame.

        Args:
            include_errors_by_pos: Whether to include position in error feature ids.
            exclude_embeddings: Whether to exclude embedding nodes.
            exclude_errors: Whether to exclude error nodes.
            exclude_logits: Whether to exclude logit nodes.
            drop_duplicates: Whether to remove duplicate nodes from df.

        Returns:
            DataFrame with feature, layer, feature_type, and ctx_idx columns.
        """
        feature_list = []
        layer_list = []
        feature_type_list = []
        ctx_idx_list = []

        for node in self.nodes:
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

            feature = self.get_feature_from_node_id(node["node_id"], deliminator="_")[1]
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
        if drop_duplicates:
            node_df = node_df.drop_duplicates(subset=GraphManager.NODE_COLUMNS)
        return node_df

    def get_frequencies_from_graph(self, node_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate frequency counts for cross layer transcoder features.

        Args:
            node_df: DataFrame of nodes. If None, creates one from graph.

        Returns:
            DataFrame with layer, feature, and ctx_freq columns.
        """
        if node_df is None:
            node_df = self.create_node_df()
        return self.get_frequencies(node_df)

    @staticmethod
    def get_frequencies(node_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate frequency counts for cross layer transcoder features.

        Args:
            node_df: DataFrame of nodes. If None, creates one from graph.

        Returns:
            DataFrame with layer, feature, and ctx_freq columns.
        """
        node_df_clts_only = node_df[node_df["feature_type"] == "cross layer transcoder"]
        ctx_freq_df = node_df_clts_only.value_counts(["layer", "feature"]).reset_index(name="ctx_freq")
        return ctx_freq_df

    def get_node_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary mapping node_id to node.

        Returns:
            Dictionary mapping node_id to node dict.
        """
        node_dict = {node["node_id"]: node for node in self.nodes}
        return node_dict

    def get_links_from_node(self, starting_node: dict = None,
                            positive_only: bool = True, hops: int = None, include_features_only: bool = False):
        """
        Find all links connected to a starting node.

        Args:
            starting_node: Node to start from. Defaults to top output logit.
            positive_only: Whether to include only positive weight edges.
            hops: Maximum number of hops from start. None for unlimited.
            include_features_only: Whether to include only cross layer transcoder features.

        Returns:
            List of link dictionaries.
        """
        allowed_nodes = {node["node_id"] for node in self.nodes
                         if not include_features_only or node["feature_type"] == 'cross layer transcoder'}
        if starting_node is None:
            starting_node = self.get_top_output_logit_node()
        starting_node_id = starting_node["node_id"]
        target_id_to_links = {}
        for link in self.links:
            target_id = link["target"]
            if target_id not in target_id_to_links:
                target_id_to_links[target_id] = []
            if not positive_only or link["weight"] > 0:
                target_id_to_links[target_id].append(link)
        relevant_links = []
        seen = set()
        new_targets = self.get_links_to_targets([starting_node_id], target_id_to_links, relevant_links, allowed_nodes)
        n_hops = 0
        while new_targets:
            n_hops += 1
            if hops and n_hops >= hops:
                break
            seen.update(new_targets)
            new_targets = self.get_links_to_targets(new_targets, target_id_to_links, relevant_links, allowed_nodes)
            new_targets = new_targets.difference(seen)

        return relevant_links

    def get_linked_sources(self, output_token_to_features: dict, positive_only: bool = True) -> bool:
        """
        Get all sources linked to each output token. Updates dictionary in place.

        Args:
            output_token_to_features: Dictionary mapping tokens to sets of node ids.
            positive_only: Whether to include only positive weight edges.

        Returns:
            True if any new sources were added, False otherwise.
        """
        newly_added = False
        for link in self.links:
            for token, nodes in output_token_to_features.items():
                if link["target"] in nodes and (not positive_only or link["weight"] > 0):
                    if link["source"] not in nodes:
                        nodes.add(link["source"])
                        newly_added = True
        return newly_added

    def get_nodes_linked_to_target(self, target_node: dict, should_sort: bool = True) -> List[dict]:
        """
        Get all nodes directly linked to a target node.

        Args:
            target_node: The target node to find links to.
            should_sort: Whether to sort by link weight descending.

        Returns:
            List of node dictionaries.
        """
        all_nodes = self.node_dict
        links = self.get_links_from_node(target_node, hops=1)
        if should_sort:
            links = sorted(links, key=lambda link: link['weight'], reverse=True)
        return [all_nodes[link['source']] for link in links]

    def get_features_linked_to_tokens(self, tok_k_outputs: int) -> Dict[str, Set[str]]:
        """
        Get all features linked to the top k output tokens.

        Args:
            tok_k_outputs: Number of top output logits to analyze.

        Returns:
            Dict mapping output tokens to sets of all linked node ids.
        """
        output_nodes = self.get_output_logits()
        top_nodes = output_nodes[:tok_k_outputs]

        output_token_to_id = {self.get_output_token_from_clerp(node): node["node_id"] for node in top_nodes}
        output_token_to_linked_features = {token: {node_id} for token, node_id in output_token_to_id.items()}

        ## Update output_token_to_linked_features with a set of linked features for each output token
        newly_added = True
        while newly_added:
            newly_added = self.get_linked_sources(output_token_to_linked_features, positive_only=True)

        return output_token_to_linked_features

    def get_top_output_logit_node(self) -> Dict[str, Any]:
        """
        Get the target output logit node.

        Returns:
            The node dict for the target output logit.
        """
        return [node for node in self.nodes if node['is_target_logit']][0]

    def get_output_logits(self) -> List[Dict[str, Any]]:
        """
        Get all output logit nodes.

        Returns:
            List of logit node dictionaries.
        """
        return [node for node in self.nodes if node['feature_type'] == "logit"]

    def find_output_node(self, node_to_find: Dict, raise_if_not_found: bool = False) -> Optional[Dict[str, Any]]:
        """
        Finds matching output node if it exists.

        Args:
            node_to_find: Find node with same id.
            raise_if_not_found: If True, raises AssertionError when no match found.

        Returns:
            The output node if it is found, None otherwise.
        """
        found_nodes = [node for node in self.get_output_logits() if
                         node['node_id'].startswith(GraphManager.get_id_without_pos(node_to_find['node_id']))]
        if not found_nodes:
            if raise_if_not_found:
                raise AssertionError(f"Can't find node corresponding with {node_to_find['clerp']} in prompt.")
            return None
        return found_nodes[0]

    @staticmethod
    def get_node_ids_from_features(feature_df: pd.DataFrame) -> list[str]:
        """
        Create node ids from a DataFrame of features.

        Args:
            feature_df: DataFrame with layer and feature columns.

        Returns:
            List of node id strings.
        """
        return [f"{feature.layer}-{feature.feature}" for feature in feature_df.itertuples()]

    @staticmethod
    def create_node_id(feature: Feature, deliminator: str = "-") -> str:
        """
        Create a node id from a Feature.

        Args:
            feature: Feature namedtuple with layer and feature.
            deliminator: Separator between layer and feature.

        Returns:
            Node id string.
        """
        return f"{feature.layer}{deliminator}{feature.feature}"

    @staticmethod
    def get_feature_from_node_id(node_id: str, deliminator: str = "-") -> Feature:
        """
        Parse a node id into a Feature.

        Args:
            node_id: Node id string to parse.
            deliminator: Separator between layer and feature.

        Returns:
            Feature namedtuple with layer and feature.
        """
        split_node_id = node_id.split(deliminator)
        return Feature(split_node_id[0], split_node_id[1])

    def get_output_token_from_clerp(self, output_node: dict = None) -> str:
        """
        Extract the output token from a node's clerp attribute.

        Args:
            output_node: Node dictionary with clerp attribute.

        Returns:
            The extracted token string.
        """
        output_node = output_node if output_node else self.get_top_output_logit_node()
        match = re.search(r'"(.*?)"', output_node["clerp"])
        if match:
            token = match.group(1).strip()
            if not token and output_node["clerp"].count("\"") > 2:
                token = '\"'
            return token
        return ''

    def check_feature_presence(self, layer: str, feature: str) -> bool:
        """
        Check if a specific feature exists in the node dataframe.
        Args:
            layer: Represents the layer.
            feature: Represents the feature id.

        Returns: True if the feature is present in the graph else False.

        """
        node_df = self.create_node_df()
        return len(node_df[(node_df['layer'] == layer) & (node_df['feature'] == feature)]) > 0

    @staticmethod
    def get_id_without_pos(node_name: str) -> str:
        """
        Remove the position suffix from a node id.

        Args:
            node_name: Node id string.

        Returns:
            Node id without position suffix.
        """
        if node_name.count("_") == 2:
            return node_name.rsplit("_", 1)[0]
        return node_name

    @staticmethod
    def get_links_to_targets(target_ids: Union[List[str], Set[str]], target_id_to_links: Dict[str, List[Dict]],
                             links: List[Dict], allowed_nodes: Set[str]) -> Set[str]:
        """
        Get links to specified target ids and collect new source targets.

        Args:
            target_ids: Set of target node ids to find links for.
            target_id_to_links: Mapping of target ids to their links.
            links: List to append found links to.
            allowed_nodes: Set of allowed source node ids.

        Returns:
            Set of new source node ids found.
        """
        for target_id in target_ids:
            linked = target_id_to_links.get(target_id, [])
            links.extend([link for link in linked if link["source"] in allowed_nodes])
        new_targets = {link["source"] for link in links}
        return new_targets
