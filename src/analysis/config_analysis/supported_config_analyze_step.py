from enum import Enum


class SupportedConfigAnalyzeStep(Enum):
    EARLY_LAYER_CONTRIBUTION = "early_layer_contribution"
    ERROR_RANKING = "error_ranking"
    REPLACEMENT_MODEL = "replacement_model"
    FEATURE_OVERLAP = "feature_overlap"
    FEATURE_PRESENCE = "feature_presence"
    SHARED_FEATURES = "shared_features"
    SUBGRAPH_FILTER = "subgraph_filter"
    TOKEN_SUBGRAPH = "token_subgraph"
