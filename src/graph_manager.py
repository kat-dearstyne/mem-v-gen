import os
from dataclasses import dataclass

import pandas as pd
import requests
from tqdm import tqdm

from common_utils import save_json, load_json, create_prompt_id, get_api_key, get_node_ids_from_features, \
    get_feature_from_node_id, get_top_output_logit_node
from constants import DEFAULT_SAVE_DIR, SAVE_FEATURE_FILENAME
from src.neuronpedia_urls import NeuronpediaUrls


@dataclass
class GraphConfig:
    model: str
    submodel: str
    max_n_logits: int = 10
    desired_logit_prob: float = 0.95
    node_threshold: float = 0.8
    edge_threshold: float = 0.85
    max_feature_nodes: int = 5000
    pruning_threshold: float = 0.8
    density_threshold: float = 0.99


class GraphManager:

    def __init__(self, graph_dir: str, config: GraphConfig):
        self.graph_dir = graph_dir
        self.config = config

    @staticmethod
    def _check_response(response: requests.Response, operation: str) -> dict:
        """
        Check response status code and return JSON if successful.

        Args:
            response: The response object from a requests call.
            operation: Description of the operation for error messages.

        Returns:
            The JSON response body.
        """
        assert response.status_code == 200, f"{operation} returned {response.status_code}"
        return response.json()

    def create_graph(self, prompt: str) -> dict:
        """
        Create an attribution graph from a text prompt using Neuronpedia API.

        Args:
            prompt: The text prompt to generate the graph from.

        Returns:
            Graph metadata dictionary.
        """
        graph_creation_response = requests.post(
            NeuronpediaUrls.GRAPH_GENERATE.value,
            headers={
                "Content-Type": "application/json"
            },
            json={
                "prompt": prompt,
                "modelId": self.config.model,
                "sourceSetName": self.config.submodel,
                "slug": "",
                "maxNLogits": self.config.max_n_logits,
                "desiredLogitProb": self.config.desired_logit_prob,
                "nodeThreshold": self.config.node_threshold,
                "edgeThreshold": self.config.edge_threshold,
                "maxFeatureNodes": self.config.max_feature_nodes
            }
        )

        graph_urls_dict = self._check_response(graph_creation_response, "Graph creation")
        graph_retrieval_response = requests.get(graph_urls_dict["s3url"])
        graph_metadata = graph_retrieval_response.json()

        url = graph_urls_dict["url"]
        graph_metadata["metadata"]['info']['url'] = url
        return graph_metadata

    def create_or_load_graph(self, prompt: str) -> dict:
        """
        Create an attribution graph or reload from file if already downloaded.

        Args:
            prompt: The text prompt to generate the graph from.

        Returns:
            Graph metadata dictionary.
        """
        prompt_id = create_prompt_id(prompt)
        graph_path = os.path.join(self.graph_dir, prompt_id, f"{self.config.model}-{self.config.submodel}.json")

        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}")
            graph_metadata = load_json(graph_path)
        else:
            print(f"Saving graph to {graph_path}")
            graph_metadata = self.create_graph(prompt=prompt)
            save_json(graph_metadata, graph_path)
        return graph_metadata

    def get_feature_from_neuronpedia(self, index: int, layer_num: int) -> dict:
        """
        Fetch feature data from the Neuronpedia API.

        Args:
            index: The feature index.
            layer_num: The layer number.

        Returns:
            Feature data dictionary.
        """
        res = requests.get(
            f"{NeuronpediaUrls.FEATURE.value}/{self.config.model}/{layer_num}-{self.config.submodel}/{index}",
            headers={
                "Content-Type": "application/json",
                "x-api-key": get_api_key()
            }
        )
        return self._check_response(res, f"Feature request {layer_num}-{index}")

    def create_feature_save_path(self, index: int | None = None, layer_num: int | None = None,
                                 base_dir: str = DEFAULT_SAVE_DIR, filename: str | None = None) -> str:
        """
        Create the save path for a given feature.

        Args:
            index: The feature index.
            layer_num: The layer number.
            base_dir: Base directory for saving.
            filename: Optional custom filename.

        Returns:
            The full save path.
        """
        if filename is None and layer_num is not None and index is not None:
            filename = SAVE_FEATURE_FILENAME.format(layer=f"{layer_num}", index=index)
        save_dir = os.path.join(base_dir, self.config.model, self.config.submodel)
        save_path = os.path.join(save_dir, filename) if filename else save_dir
        return save_path

    def save_feature(self, feature_json: dict, base_dir: str = DEFAULT_SAVE_DIR,
                     filename: str | None = None) -> str:
        """
        Save feature data to a JSON file.

        Args:
            feature_json: The feature data dictionary.
            base_dir: Base directory for saving.
            filename: Optional custom filename.

        Returns:
            The save path.
        """
        layer_num, submodel = feature_json["layer"].split("-", 1)
        save_path = self.create_feature_save_path(
            index=feature_json["index"], layer_num=int(layer_num),
            base_dir=base_dir, filename=filename
        )
        save_json(feature_json, save_path)
        return save_path

    def reload_feature(self, index: int, layer_num: int,
                       base_dir: str = DEFAULT_SAVE_DIR, filename: str | None = None) -> dict:
        """
        Load previously saved feature data from a JSON file.

        Args:
            index: The feature index.
            layer_num: The layer number.
            base_dir: Base directory for loading.
            filename: Optional custom filename.

        Returns:
            Feature data dictionary.
        """
        save_path = self.create_feature_save_path(
            index=index, layer_num=layer_num, base_dir=base_dir, filename=filename
        )
        return load_json(save_path)

    def get_feature(self, index: int, layer_num: int,
                    base_dir: str = DEFAULT_SAVE_DIR, filename: str | None = None) -> dict:
        """
        Get feature data from cache or fetch from Neuronpedia if not cached.

        Args:
            index: The feature index.
            layer_num: The layer number.
            base_dir: Base directory for caching.
            filename: Optional custom filename.

        Returns:
            Feature data dictionary.
        """
        save_path = self.create_feature_save_path(
            index=index, layer_num=layer_num, base_dir=base_dir, filename=filename
        )
        if os.path.exists(save_path):
            return self.reload_feature(index=index, layer_num=layer_num, base_dir=base_dir, filename=filename)
        else:
            feature_json = self.get_feature_from_neuronpedia(index=index, layer_num=layer_num)
            self.save_feature(feature_json, base_dir=base_dir, filename=filename)
            return feature_json

    def get_all_features(self, features_df: pd.DataFrame,
                         base_dir: str = DEFAULT_SAVE_DIR, filename: str | None = None) -> list[dict]:
        """
        Retrieve feature data for all features in a DataFrame.

        Args:
            features_df: DataFrame containing features with 'layer' and 'feature' columns.
            base_dir: Base directory for caching.
            filename: Optional custom filename.

        Returns:
            List of feature data dictionaries.
        """
        print(f"Saving features to {self.create_feature_save_path(base_dir=base_dir)}")
        features = []
        for feature_row in tqdm(features_df.itertuples(), desc="Getting features from attribution graph",
                                total=len(features_df)):
            feature = self.get_feature(
                index=feature_row.feature, layer_num=feature_row.layer,
                base_dir=base_dir, filename=filename
            )
            features.append(feature)
        return features

    def create_feature_list(self, prompt_id: str) -> str | None:
        """
        Create or retrieve a feature list for the given prompt ID.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            The list ID.
        """
        list_name = f"{prompt_id}:{self.config.model}:{self.config.submodel}"
        api_key = get_api_key()
        res_lists = requests.post(
            NeuronpediaUrls.LIST_LIST.value,
            headers={
                "x-api-key": api_key
            }
        )
        lists_data = self._check_response(res_lists, "Getting lists")
        list_ids = [item["id"] for item in lists_data if item["name"] == list_name]
        if list_ids:
            list_id = list_ids[0]
        else:
            res_list_create = requests.post(
                NeuronpediaUrls.LIST_NEW.value,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key
                },
                json={
                    "name": list_name,
                    "description": "Top frequency features for prompt",
                    "testText": None
                }
            )
            list_id = self._check_response(res_list_create, "Creating new list")["id"]
        return list_id

    def add_features_to_list(self, features_df: pd.DataFrame, prompt_id: str) -> str:
        """
        Add features to a Neuronpedia list.

        Args:
            features_df: DataFrame containing features with 'layer' and 'feature' columns.
            prompt_id: The prompt identifier.

        Returns:
            The list URL.
        """
        list_id = self.create_feature_list(prompt_id=prompt_id)
        api_key = get_api_key()
        res_list_info = requests.post(
            NeuronpediaUrls.LIST_GET.value,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key
            },
            json={
                "listId": list_id
            }
        )
        list_info = self._check_response(res_list_info, "Getting list")
        existing_features = {(neuron["modelId"], neuron["layer"], neuron["index"])
                            for neuron in list_info["neurons"]}
        features_to_add = [
            {
                "modelId": self.config.model,
                "layer": f"{feature.layer}-{self.config.submodel}",
                "index": feature.feature,
                "description": "Top frequency features for prompt"
            }
            for feature in features_df.itertuples()
            if (self.config.model, f"{feature.layer}-{self.config.submodel}", feature.feature) not in existing_features
        ]
        if features_to_add:
            res_list_add = requests.post(
                NeuronpediaUrls.LIST_ADD_FEATURES.value,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key
                },
                json={
                    "listId": list_id,
                    "featuresToAdd": features_to_add
                }
            )
            self._check_response(res_list_add, "Saving to list")
        return f"{NeuronpediaUrls.BASE.value}/list/{list_id}"

    def _create_subgraph(self, graph_metadata: dict, pinned_ids: list[str],
                         list_name: str = "top features") -> str:
        """
        Create a subgraph on Neuronpedia with the given pinned node IDs.

        Args:
            graph_metadata: The graph metadata dictionary.
            pinned_ids: List of node IDs to pin in the subgraph.
            list_name: Display name for the subgraph.

        Returns:
            The subgraph ID.
        """
        res = requests.post(
            NeuronpediaUrls.SUBGRAPH_SAVE.value,
            headers={
                "Content-Type": "application/json",
                "x-api-key": get_api_key()
            },
            json={
                "modelId": graph_metadata["metadata"]["scan"],
                "slug": graph_metadata["metadata"]["slug"],
                "displayName": list_name,
                "pinnedIds": pinned_ids,
                "supernodes": [],
                "clerps": [],
                "pruningThreshold": self.config.pruning_threshold,
                "densityThreshold": self.config.density_threshold,
                "overwriteId": ""
            }
        )
        return self._check_response(res, "Creating subgraph")['subgraphId']

    def create_subgraph_from_selected_features(self, feature_df: pd.DataFrame, graph_metadata: dict,
                                               list_name: str = "top features", include_output_node: bool = True) -> str:
        """
        Create a subgraph from selected features.

        Args:
            feature_df: DataFrame containing features with 'layer' and 'feature' columns.
            graph_metadata: The graph metadata dictionary.
            list_name: Display name for the subgraph.
            include_output_node: Whether to include the target output logit node.

        Returns:
            The subgraph ID.
        """
        node_ids = get_node_ids_from_features(feature_df)
        selected_features = {node_id: [] for node_id in node_ids}
        graph_nodes = graph_metadata["nodes"]
        for node in graph_nodes:
            feature_id = get_feature_from_node_id(node["node_id"], deliminator="_")[1]
            feature_key = f"{node['layer']}-{feature_id}"
            if feature_key in selected_features:
                selected_features[feature_key].append(node["node_id"])

        pinned_ids = [node for nodes in selected_features.values() for node in nodes]
        if include_output_node:
            pinned_ids.append(get_top_output_logit_node(graph_metadata["nodes"])["node_id"])

        return self._create_subgraph(graph_metadata, pinned_ids, list_name)

    def create_subgraph_from_links(self, links: pd.DataFrame, graph_metadata: dict,
                                   list_name: str = "top features") -> str:
        """
        Create a subgraph from graph links.

        Args:
            links: DataFrame containing links with 'source' and 'target' columns.
            graph_metadata: The graph metadata dictionary.
            list_name: Display name for the subgraph.

        Returns:
            The subgraph ID.
        """
        node_ids = {link['target'] for link in links}
        node_ids.update({link['source'] for link in links})
        return self._create_subgraph(graph_metadata, list(node_ids), list_name)

    def get_subgraphs(self, graph_metadata: dict) -> dict:
        """
        Retrieve all subgraphs associated with a graph.

        Args:
            graph_metadata: The graph metadata dictionary.

        Returns:
            Dictionary containing subgraph information.
        """
        res = requests.post(
            NeuronpediaUrls.SUBGRAPH_LIST.value,
            headers={
                "Content-Type": "application/json",
                "x-api-key": get_api_key()
            },
            json={
                "modelId": graph_metadata["metadata"]["scan"],
                "slug": graph_metadata["metadata"]["slug"],
            }
        )
        return self._check_response(res, "Getting subgraphs")

