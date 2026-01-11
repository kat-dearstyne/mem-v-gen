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


class EarlyLayerMetrics(Metrics):
    """Metrics for early layer contribution analysis."""
    MAX_LAYER = "max_layer"
    EARLY_LAYER_FRACTION = "early_layer_fraction"


class SignificanceMetrics(Metrics):
    """Metrics for pairwise significance testing."""
    GROUP1_MEAN = "group1_mean"
    GROUP2_MEAN = "group2_mean"
    GROUP1_STD = "group1_std"
    GROUP2_STD = "group2_std"
    N_PER_GROUP = "n_per_group"
    T_STATISTIC = "t_statistic"
    T_P_VALUE = "t_p_value"
    T_SIGNIFICANT = "t_significant"
    MANN_WHITNEY_U = "mann_whitney_u"
    MW_P_VALUE = "mw_p_value"
    MW_SIGNIFICANT = "mw_significant"
    COHENS_D = "cohens_d"
    RANK_BISERIAL_R = "rank_biserial_r"


class OmnibusSignificanceMetrics(Metrics):
    """Metrics for omnibus significance testing (ANOVA/Kruskal-Wallis)."""
    F_STATISTIC = "f_statistic"
    ANOVA_P_VALUE = "anova_p_value"
    ANOVA_SIGNIFICANT = "anova_significant"
    ETA_SQUARED = "eta_squared"
    H_STATISTIC = "h_statistic"
    KRUSKAL_P_VALUE = "kruskal_p_value"
    KRUSKAL_SIGNIFICANT = "kruskal_significant"
    EPSILON_SQUARED = "epsilon_squared"


class PooledStatsMetrics(Metrics):
    """Metrics for pooled error ranking statistics."""
    MEAN = "mean"
    MEDIAN = "median"
    WILCOXON_STAT_ONESIDED = "wilcoxon_stat_onesided"
    WILCOXON_P_ONESIDED = "wilcoxon_p_onesided"
    COUNT_BASE_GT_OTHER = "count_base_gt_other"
    COUNT_BASE_LT_OTHER = "count_base_lt_other"
    COUNT_EQUAL = "count_equal"
    N_SAMPLES = "n_samples"


class ConditionStatsMetrics(Metrics):
    """Metrics for condition-level error ranking statistics."""
    MEAN = "mean"
    MEDIAN = "median"
    WILCOXON_STAT = "wilcoxon_stat"
    WILCOXON_P = "wilcoxon_p"
    TTEST_STAT = "ttest_stat"
    TTEST_P = "ttest_p"
    COUNT_G1_GT_G2 = "count_g1_gt_g2"
    COUNT_G1_LT_G2 = "count_g1_lt_g2"
    COUNT_EQUAL = "count_equal"


class RawScoreMetrics(Metrics):
    """Metrics for raw error ranking scores."""
    METRIC = "metric"
    SAMPLE = "sample"
    G1_SCORE = "g1_score"
    G2_SCORE = "g2_score"
    DELTA = "delta"
