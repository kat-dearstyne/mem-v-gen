from enum import Enum


class SupportedConfigAnalyzeStep(Enum):
    ERROR_RANKING = "error_ranking"
    REPLACEMENT_MODEL = "replacement_model"
    OVERLAP = "overlap"
    FILTER = "filter"
    SIM = "sim"
    FEATURE_PRESENCE = "feature_presence"
    SHARED_FEATURES = "shared_features"
