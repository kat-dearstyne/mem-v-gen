import os
import uuid
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.constants import DEFAULT_SAVE_DIR, SAVE_FEATURE_FILENAME
from src.graph_manager import GraphManager
from src.neuronpedia_urls import NeuronpediaUrls
from src.utils import load_json, save_json, get_api_key, create_run_uuid


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


class NeuronpediaManager:

    def __init__(self, graph_dir: str, config: GraphConfig, feature_dir: str = DEFAULT_SAVE_DIR):
        """
        Manages retrieving, loading and saving data from Neuronpedia.

        Args:
            graph_dir: Directory to save graphs.
            config: GraphConfig instance with model settings.
            feature_dir: Directory to save features.
        """
        self.graph_dir = graph_dir
        self.config = config
        self.feature_dir = feature_dir

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

    def create_graph(self, prompt: str) -> GraphManager:
        """
        Create an attribution graph from a text prompt using Neuronpedia API.

        Args:
            prompt: The text prompt to generate the graph from.

        Returns:
            GraphManager instance wrapping the graph metadata.
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
        return GraphManager(graph_metadata)

    def create_or_load_graph(self, prompt: str) -> GraphManager:
        """
        Create an attribution graph or reload from file if already downloaded.

        Args:
            prompt: The text prompt to generate the graph from.

        Returns:
            GraphManager instance wrapping the graph metadata.
        """
        prompt_id = self.create_prompt_id(prompt)
        graph_path = os.path.join(self.graph_dir, prompt_id, f"{self.config.model}-{self.config.submodel}.json")

        if os.path.exists(graph_path):
            print(f"Loading graph from {graph_path}")
            graph_metadata = load_json(graph_path)
        else:
            print(f"Saving graph to {graph_path}")
            graph_manager = self.create_graph(prompt=prompt)
            save_json(graph_manager.graph_metadata, graph_path)
            return graph_manager
        return GraphManager(graph_metadata)

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

    def _create_feature_save_path(self, index: int | None = None, layer_num: int | None = None) -> str:
        """
        Create the save path for a given feature.

        Args:
            index: The feature index.
            layer_num: The layer number.

        Returns:
            The full save path.
        """
        save_dir = os.path.join(self.feature_dir, self.config.model, self.config.submodel)
        if layer_num is not None and index is not None:
            filename = SAVE_FEATURE_FILENAME.format(layer=f"{layer_num}", index=index)
            return os.path.join(save_dir, filename)
        return save_dir

    def _save_feature(self, feature_json: dict) -> str:
        """
        Save feature data to a JSON file.

        Args:
            feature_json: The feature data dictionary.

        Returns:
            The save path.
        """
        layer_num, _ = feature_json["layer"].split("-", 1)
        save_path = self._create_feature_save_path(
            index=feature_json["index"], layer_num=int(layer_num)
        )
        save_json(feature_json, save_path)
        return save_path

    def _reload_feature(self, index: int, layer_num: int) -> dict:
        """
        Load previously saved feature data from a JSON file.

        Args:
            index: The feature index.
            layer_num: The layer number.

        Returns:
            Feature data dictionary.
        """
        save_path = self._create_feature_save_path(index=index, layer_num=layer_num)
        return load_json(save_path)

    def get_feature(self, index: int, layer_num: int) -> dict:
        """
        Get feature data from cache or fetch from Neuronpedia if not cached.

        Args:
            index: The feature index.
            layer_num: The layer number.

        Returns:
            Feature data dictionary.
        """
        save_path = self._create_feature_save_path(index=index, layer_num=layer_num)
        if os.path.exists(save_path):
            return self._reload_feature(index=index, layer_num=layer_num)
        else:
            feature_json = self.get_feature_from_neuronpedia(index=index, layer_num=layer_num)
            self._save_feature(feature_json)
            return feature_json

    def get_all_features(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Retrieve feature data for all features in a DataFrame.

        Args:
            features_df: DataFrame containing features with 'layer' and 'feature' columns.

        Returns:
            List of feature data dictionaries.
        """
        print(f"Saving features to {self._create_feature_save_path()}")
        features = []
        for feature_row in tqdm(features_df.itertuples(), desc="Getting features from attribution graph",
                                total=len(features_df)):
            feature = self.get_feature(index=feature_row.feature, layer_num=feature_row.layer)
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

    def _create_subgraph(self, graph: GraphManager, pinned_ids: list[str],
                         list_name: str = "top features") -> str:
        """
        Create a subgraph on Neuronpedia with the given pinned node IDs.

        Args:
            graph: GraphManager instance containing the graph.
            pinned_ids: List of node IDs to pin in the subgraph.
            list_name: Display name for the subgraph.

        Returns:
            The subgraph ID.
        """
        list_name +=  f"{create_run_uuid()}"
        res = requests.post(
            NeuronpediaUrls.SUBGRAPH_SAVE.value,
            headers={
                "Content-Type": "application/json",
                "x-api-key": get_api_key()
            },
            json={
                "modelId": graph.metadata["scan"],
                "slug": graph.metadata["slug"],
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

    def create_subgraph_from_selected_features(self, feature_df: pd.DataFrame, graph: GraphManager,
                                               list_name: str = "top features",
                                               include_output_node: bool = True) -> str:
        """
        Create a subgraph from selected features.

        Args:
            feature_df: DataFrame containing features with 'layer' and 'feature' columns.
            graph: GraphManager instance containing the graph.
            list_name: Display name for the subgraph.
            include_output_node: Whether to include the target output logit node.

        Returns:
            The subgraph ID.
        """
        node_ids = GraphManager.get_node_ids_from_features(feature_df)
        selected_features = {node_id: [] for node_id in node_ids}
        for node in graph.nodes:
            feature_id = GraphManager.get_feature_from_node_id(node["node_id"], deliminator="_")[1]
            feature_key = f"{node['layer']}-{feature_id}"
            if feature_key in selected_features:
                selected_features[feature_key].append(node["node_id"])

        pinned_ids = [node for nodes in selected_features.values() for node in nodes]
        if include_output_node:
            pinned_ids.append(graph.get_top_output_logit_node()["node_id"])

        return self._create_subgraph(graph, pinned_ids, list_name)

    def create_subgraph_from_links(self, links: pd.DataFrame, graph: GraphManager,
                                   list_name: str = "top features") -> str:
        """
        Create a subgraph from graph links.

        Args:
            links: DataFrame containing links with 'source' and 'target' columns.
            graph: GraphManager instance containing the graph.
            list_name: Display name for the subgraph.

        Returns:
            The subgraph ID.
        """
        node_ids = {link['target'] for link in links}
        node_ids.update({link['source'] for link in links})
        return self._create_subgraph(graph, list(node_ids), list_name)

    def get_subgraphs(self, graph: GraphManager) -> dict:
        """
        Retrieve all subgraphs associated with a graph.

        Args:
            graph: GraphManager instance containing the graph.

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
                "modelId": graph.metadata["scan"],
                "slug": graph.metadata["slug"],
            }
        )
        return self._check_response(res, "Getting subgraphs")

    def filter_features_for_subgraph(self, features_df: pd.DataFrame,
                                     graph_manager: GraphManager = None,
                                     filter_by_act_density: Optional[int] = None,
                                     filter_layers_less_than: int = None) -> pd.DataFrame:
        """
        Filters features by excluding embedding/output layers and optionally by activation density.

        Args:
            features_df: DataFrame containing features with 'layer' and 'feature' columns.
            graph_manager: Optional GraphManager to exclude output layer nodes.
            filter_by_act_density: Optional threshold to exclude features with activation density above this percentage.
            filter_layers_less_than: Optional layer number to exclude all layers below.

        Returns:
            Filtered DataFrame with selected features.
        """
        filter_layers = ['E']
        if graph_manager is not None:
            output_node = graph_manager.get_top_output_logit_node()
            filter_layers.append(output_node['layer'])  # exclude all output nodes

        if filter_layers_less_than is not None:  # exclude all layers less than given layer number
            filter_layers.extend([str(i) for i in range(filter_layers_less_than)])

        filtered = features_df[~features_df['layer'].isin(filter_layers)]

        if filter_by_act_density and len(filtered) > 0:  # remove features with high activation density
            feature_info = self.get_all_features(filtered)
            selected_features = [
                feature_row for info, feature_row in zip(feature_info, filtered.itertuples())
                if info['frac_nonzero'] * 100 < filter_by_act_density and info['frac_nonzero'] > 0
            ]
            filtered = pd.DataFrame(selected_features)

        if len(filtered) > 0:
            filtered = filtered[GraphManager.NODE_COLUMNS] \
                if 'ctx_freq' not in filtered.columns else filtered[GraphManager.NODE_COLUMNS + ['ctx_freq']]

        return filtered

    @staticmethod
    def create_prompt_id(prompt: str) -> str:
        """
        Creates a unique prompt id for a specified prompt.


        Args:
            prompt: The prompt to generate an ID for.

        Returns:
            ID for the prompt.
        """
        prompt_start = "_".join(prompt.lower().split()[:5])
        prompt_id = f"{prompt_start}:{uuid.uuid5(uuid.NAMESPACE_DNS, prompt)}"
        return prompt_id
