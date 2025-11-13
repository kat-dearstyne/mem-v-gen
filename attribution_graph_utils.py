import os
from typing import Optional, List, Tuple, Set, Any

import pandas as pd
import requests
from pandas import DataFrame
from tqdm import tqdm

from constants import DEFAULT_SAVE_DIR, SAVE_FEATURE_FILENAME
from common_utils import save_json, load_json, get_api_key, create_prompt_id, get_node_ids_from_features, \
    get_feature_from_node_id, Feature, create_node_id


def create_graph(prompt: str, model: str, submodel: str) -> dict:
    """
    Create an attribution graph from a text prompt using Neuronpedia API.
    """
    graph_creation_response = requests.post(
        "https://www.neuronpedia.org/api/graph/generate",
        headers={
            "Content-Type": "application/json"
        },
        json={
            "prompt": prompt,
            "modelId": model,
            "sourceSetName": submodel,
            "slug": "",
            "maxNLogits": 10,
            "desiredLogitProb": 0.95,
            "nodeThreshold": 0.8,
            "edgeThreshold": 0.85,
            "maxFeatureNodes": 5000
        }
    )

    assert graph_creation_response.status_code == 200, \
        f"Graph creation response returned {graph_creation_response.status_code}"
    graph_urls_dict = graph_creation_response.json()
    graph_retrieval_response = requests.get(graph_urls_dict["s3url"])
    graph_metadata = graph_retrieval_response.json()

    url = graph_urls_dict["url"]
    graph_metadata["metadata"]['info']['url'] = url
    return graph_metadata


def create_or_load_graph(graph_dir: str, model: str, submodel: str, prompt: str) -> dict:
    """
    Create an attribution graph from a text prompt using Neuronpedia API or reloads from file if already downloaded.
    """
    prompt_id = create_prompt_id(prompt)
    graph_path = os.path.join(graph_dir, prompt_id, f"{model}-{submodel}.json")

    if os.path.exists(graph_path):
        print(f"Loading graph from {graph_path}")
        graph_metadata = load_json(graph_path)
    else:
        print(f"Saving graph to {graph_path}")
        graph_metadata = create_graph(prompt=prompt, model=model, submodel=submodel)
        save_json(graph_metadata, graph_path)
    return graph_metadata


def create_node_df(graph_metadata: dict) -> pd.DataFrame:
    """
    Extract node information from graph metadata into a DataFrame.
    """
    feature_list = []
    layer_list = []
    feature_type_list = []
    ctx_idx_list = []

    for node in graph_metadata["nodes"]:
        feature = get_feature_from_node_id(node["node_id"], deliminator="_")[1]
        feature_list.append(feature)

        layer = node["layer"]
        layer_list.append(layer)

        feature_type = node["feature_type"]
        feature_type_list.append(feature_type)

        ctx_idx = node["ctx_idx"]
        ctx_idx_list.append(ctx_idx)

    node_df = pd.DataFrame({
        "feature": feature_list,
        "layer": layer_list,
        "feature_type": feature_type_list,
        "ctx_idx": ctx_idx_list
    })
    return node_df


def get_frequencies(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate frequency counts for features.
    """
    node_df_clts_only = node_df[node_df["feature_type"] == "cross layer transcoder"]
    ctx_freq_df = node_df_clts_only.value_counts(["layer", "feature"]).reset_index(name="ctx_freq")
    return ctx_freq_df


def create_feature_save_path(index: int = None, layer_num: int = None, model: str = "gemma-2-2b",
                             submodel: str = "gemmascope-transcoder-16k",
                             base_dir: Optional[str] = DEFAULT_SAVE_DIR, filename: Optional[str] = None) -> str:
    """
    Creates the save path for a given feature.
    """
    if filename is None and layer_num and index:
        filename = SAVE_FEATURE_FILENAME.format(layer=f"{layer_num}", index=index)
    save_dir = os.path.join(base_dir, model, submodel)
    save_path = os.path.join(save_dir, filename) if filename else save_dir
    return save_path


def save_feature(feature_json: dict, foldername: Optional[str] = DEFAULT_SAVE_DIR,
                 filename: Optional[str] = None) -> str:
    """
    Save feature data to a JSON file in the specified directory.
    """
    layer_num, submodel = feature_json["layer"].split("-", 1)
    save_path = create_feature_save_path(index=feature_json["index"], layer_num=layer_num,
                                         model=feature_json["modelId"],
                                         submodel=submodel, filename=filename, base_dir=foldername)
    save_json(feature_json, save_path)
    return save_path


def reload_feature(index: int, layer_num: int, model: str, submodel: str,
                   foldername: Optional[str] = DEFAULT_SAVE_DIR, filename: Optional[str] = None) -> dict:
    """
    Load previously saved feature data from a JSON file.
    """
    save_path = create_feature_save_path(index=index, layer_num=layer_num, model=model, submodel=submodel,
                                         filename=filename, base_dir=foldername)
    feature_json = load_json(save_path)
    return feature_json


def get_feature_from_neuronpedia(index: int, layer_num: int, model: str,
                                 submodel: str) -> dict:
    """
    Fetch feature data directly from the Neuronpedia API.
    """
    res = requests.get(
        f"https://www.neuronpedia.org/api/feature/{model}/{layer_num}-{submodel}/{index}",
        headers={
            "Content-Type": "application/json",
            "x-api-key": get_api_key()
        }
    )
    assert res.status_code == 200, f"Requested returned {res.status_code}"
    return res.json()


def get_feature(index: int, layer_num: int, model: str, submodel: str,
                foldername: Optional[str] = DEFAULT_SAVE_DIR, filename: Optional[str] = None) -> dict:
    """
    Get feature data from cache or fetch from Neuronpedia if not cached.
    """
    if os.path.exists(create_feature_save_path(index=index, layer_num=layer_num, model=model, submodel=submodel,
                                               base_dir=foldername, filename=filename)):
        feature_json = reload_feature(index=index, layer_num=layer_num, model=model, submodel=submodel,
                                      foldername=foldername, filename=filename)
    else:
        feature_json = get_feature_from_neuronpedia(index=index, layer_num=layer_num, model=model, submodel=submodel)
        save_feature(feature_json, foldername=foldername, filename=filename)
    return feature_json


def get_all_features(features_df: pd.DataFrame, model: str, submodel: str,
                     foldername: Optional[str] = DEFAULT_SAVE_DIR, filename: Optional[str] = None) -> list[dict]:
    """
    Retrieve feature data for all features in the DataFrame.
    """
    print(f"Saving features to {create_feature_save_path(model=model, submodel=submodel, base_dir=foldername)}")
    features = []
    for feature_row in tqdm(features_df.itertuples(), desc="Getting features from attribution graph",
                            total=len(features_df)):
        feature = get_feature(index=feature_row.feature, layer_num=feature_row.layer,
                              model=model, submodel=submodel, foldername=foldername, filename=filename)
        features.append(feature)
    return features


def add_features_to_list(features_df: pd.DataFrame, prompt_id: str, model: str,
                         submodel: str) -> str:
    """
    Add features to a Neuronpedia list and return the list URL.
    """
    list_id = create_feature_list(prompt_id=prompt_id, model=model, submodel=submodel)
    api_key = get_api_key()
    res_list_info = requests.post("https://www.neuronpedia.org/api/list/get",
                                  headers={
                                      "Content-Type": "application/json",
                                      "x-api-key": api_key
                                  },
                                  json={
                                      "listId": list_id
                                  }
                                  )
    assert res_list_info.status_code == 200, f"Getting list returned response {res_list_info.status_code}"
    existing_features = {(neuron["modelId"], neuron["layer"], neuron["index"])
                         for neuron in res_list_info.json()["neurons"]}
    features_to_add = [
        {
            "modelId": model,
            "layer": f"{feature.layer}-{submodel}",
            "index": feature.feature,
            "description": "Top frequency features for prompt"
        }
        for feature in features_df.itertuples()
        if (model, f"{feature.layer}-{submodel}", feature.feature) not in existing_features
    ]
    if features_to_add:
        res_list_add = requests.post(
            "https://www.neuronpedia.org/api/list/add-features",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key
            },
            json={
                "listId": list_id,
                "featuresToAdd": features_to_add
            }
        )
        assert res_list_add.status_code == 200, f"Saving to list returned response {res_list_add.status_code}"
    return f"https://www.neuronpedia.org/list/{list_id}"


def create_feature_list(prompt_id: str, model: str, submodel: str) -> str | None:
    """
    Create or retrieve a feature list for the given prompt ID.
    """
    list_name = f"{prompt_id}:{model}:{submodel}"
    api_key = get_api_key()
    res_lists = requests.post(
        "https://www.neuronpedia.org/api/list/list",
        headers={
            "x-api-key": api_key
        }
    )
    assert res_lists.status_code == 200, f"Getting lists returned response {res_lists.status_code}"
    lists_data = res_lists.json()
    list_ids = [item["id"] for item in lists_data if item["name"] == list_name]
    if list_ids:
        list_id = list_ids[0]
    else:
        res_list_create = requests.post("https://www.neuronpedia.org/api/list/new",
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
        assert res_list_create.status_code == 200, f"Creating new list returned response {res_list_create.status_code}"
        list_id = res_list_create.json()["id"]
    return list_id


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


def create_subgraph_from_selected_features(feature_df: pd.DataFrame, graph_metadata: dict,
                                           list_name: str = "top features", include_output_node: bool = True) -> str:
    """
    Creates a subgraph of the selected features on Neuronpedia.
    """
    node_ids = get_node_ids_from_features(feature_df)
    selected_features = {node_id: [] for node_id in node_ids}
    graph_nodes = graph_metadata["nodes"]
    for node in graph_nodes:
        feature_id = get_feature_from_node_id(node["node_id"], deliminator="_")[1]
        feature_key =  f"{node['layer']}-{feature_id}"
        if feature_key in selected_features:
            selected_features[feature_key].append(node["node_id"])
    output_nodes = []
    if include_output_node:
        output_nodes.append(get_output_logit_node(graph_metadata["nodes"])["node_id"])

    res = requests.post(
        "https://www.neuronpedia.org/api/graph/subgraph/save",
        headers={
            "Content-Type": "application/json",
            "x-api-key": get_api_key()
        },
        json={
            "modelId": graph_metadata["metadata"]["scan"],
            "slug": graph_metadata["metadata"]["slug"],
            "displayName": list_name,
            "pinnedIds": [node for nodes in selected_features.values() for node in nodes] + output_nodes,
            "supernodes": [[name] + nodes for name, nodes in selected_features.items()],
            "clerps": [],
            "pruningThreshold": 0.8,
            "densityThreshold": 0.99,
            "overwriteId": ""
        }
    )
    res_json = res.json()
    return res_json['subgraphId']


def get_output_logit_node(nodes: List[dict]) -> Any:
    """
    Gets the the node that represents the target output logit from a list of all graph nodes.
    """
    return [node for node in nodes if node['is_target_logit']][0]


def get_subgraphs(graph_metadata) -> dict:
    """
    Creates a subgraph of the selected features on Neuronpedia.
    """
    res = requests.post(
        "https://www.neuronpedia.org/api/graph/subgraph/list",
        headers={
            "Content-Type": "application/json",
            "x-api-key": get_api_key()
        },
        json={
            "modelId": graph_metadata["metadata"]["scan"],
            "slug": graph_metadata["metadata"]["slug"],
        }
    )
    res_json = res.json()
    return res_json

def get_linked_sources(graph_metadata: dict, output_token_to_features: dict, positive_only: bool = True):
    """
    Gets all sources linked to each output token.
    """
    newly_added = False
    for link in graph_metadata["links"]:
        for token, nodes in output_token_to_features.items():
            if link["target"] in nodes and (not positive_only or link["weight"] > 0):
                if link["source"] not in nodes:
                    nodes.add(link["source"])
                    newly_added = True
    return newly_added