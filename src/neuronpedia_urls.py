from enum import Enum


class NeuronpediaUrls(Enum):
    BASE = "https://www.neuronpedia.org"
    GRAPH_GENERATE = "https://www.neuronpedia.org/api/graph/generate"
    FEATURE = "https://www.neuronpedia.org/api/feature"
    LIST_GET = "https://www.neuronpedia.org/api/list/get"
    LIST_ADD_FEATURES = "https://www.neuronpedia.org/api/list/add-features"
    LIST_LIST = "https://www.neuronpedia.org/api/list/list"
    LIST_NEW = "https://www.neuronpedia.org/api/list/new"
    SUBGRAPH_SAVE = "https://www.neuronpedia.org/api/graph/subgraph/save"
    SUBGRAPH_LIST = "https://www.neuronpedia.org/api/graph/subgraph/list"
