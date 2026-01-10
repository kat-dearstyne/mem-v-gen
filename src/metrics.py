from enum import Enum


class Metrics(Enum):
    """Base enum class for metric definitions with display formatting."""

    def get_printable(self) -> str:
        """
        Returns the name with spaces instead of underscores.

        Returns:
            The metric name formatted with spaces and title case.
        """
        return " ".join(self.value.split("_")).title()


class ComparisonMetrics(Metrics):
    JACCARD_INDEX = 'jaccard_index'
    WEIGHTED_JACCARD = 'weighted_jaccard'
    FRAC_FROM_INTERSECTION = 'frac_from_intersection'
    SHARED_TOKEN = 'shared_token'
    OUTPUT_PROBABILITY = 'output_probability'


class SharedFeatureMetrics(Metrics):
    NUM_SHARED = 'num_shared'
    NUM_PROMPTS = 'num_prompts'
    AVG_FEATURES_PER_PROMPT = 'avg_features_per_prompt'
    COUNT_AT_THRESHOLD = 'count_at_{}pct'
    SHARED_PRESENT_PER_PROMPT = 'shared_present_per_prompt'


class SubgraphComparisonMetrics(Metrics):
    """Top-level categories for subgraph comparison results."""
    INTERSECTION_METRICS = 'intersection_metrics'
    FEATURE_PRESENCE = 'feature_presence'
    SHARED_FEATURES = 'shared_features'


class FeatureSharingMetrics(Metrics):
    """Metrics for feature and edge sharing analysis."""
    MAIN_TOTAL_WEIGHT = 'main_total_weight'
    UNIQUE_WEIGHT = 'unique_weight'
    SHARED_WEIGHT = 'shared_weight'
    UNIQUE_WEIGHTED_FRAC = 'unique_weighted_frac'
    SHARED_WEIGHTED_FRAC = 'shared_weighted_frac'
    UNIQUE_FRAC = 'unique_frac'
    SHARED_FRAC = 'shared_frac'
    NUM_UNIQUE = 'num_unique'
    NUM_SHARED = 'num_shared'
    NUM_MAIN = 'num_main'


class ErrorRankingMetrics(Metrics):
    MANN_WHITNEY = "mann_whitney"
    TOP_K = "top_k_error_proportion"
    NDCG = "ndcg"
    AP = "average_precision"


class ReplacementAccuracyMetrics(Metrics):
    """Metrics for comparing base model and replacement model outputs."""
    LAST_TOKEN_COSINE = "last_token_cosine"
    CUMULATIVE_COSINE = "cumulative_cosine"
    ORIGINAL_ACCURACY = "original_accuracy"
    ORIGINAL_TOP_TOKEN = "original_top_token"
    REPLACEMENT_TOP_TOKEN = "replacement_top_token"
    KL_DIVERGENCE = "kl_divergence"
    TOP_K_AGREEMENT = "top_k_agreement"
    REPLACEMENT_PROB_OF_ORIGINAL_TOP = "replacement_prob_of_original_top"
    PERPLEXITY_LAST_TOKEN = "perplexity_last_token"
    PERPLEXITY_FULL = "perplexity_full"
    PER_POSITION_COSINE = "per_position_cosine"
    PER_POSITION_KL = "per_position_kl"
    PER_POSITION_ARGMAX_MATCH = "per_position_argmax_match"
    PER_POSITION_CROSS_ENTROPY = "per_position_cross_entropy"


class ComplexityMetrics(Metrics):
    """Metrics for measuring token complexity."""
    ZIPF_FREQUENCY = "zipf_frequency"
    TOKEN_LENGTH = "token_length"
