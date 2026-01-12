import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.analysis.cross_config_analysis.cross_config_analyzer import CrossConfigAnalyzer, STEP2CLASS
from src.analysis.cross_config_analysis.cross_config_error_ranking_step import (
    CrossConfigErrorRankingStep
)
from src.analysis.cross_config_analysis.cross_config_replacement_model_accuracy_step import (
    CrossConfigReplacementModelAccuracyStep
)
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CrossConfigSubgraphFilterStep, DIFF_KEY, SIM_KEY, SHARED_FEATURES_KEY,
    CONFIG_NAME_COL, PROMPT_TYPE_COL
)
from src.analysis.cross_config_analysis.cross_config_feature_overlap_step import (
    CrossConfigFeatureOverlapStep, CONFIG_NAME_COL as FEATURE_CONFIG_NAME_COL
)
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.config_analysis.config_error_ranking_step import PermutationTestResult
from src.metrics import (
    ErrorRankingMetrics, ReplacementAccuracyMetrics, ComplexityMetrics,
    ComparisonMetrics, SharedFeatureMetrics, FeatureSharingMetrics
)


class TestCrossConfigAnalyzerFixtures:
    """Shared test fixtures for CrossConfigAnalyzer tests."""

    @staticmethod
    def create_error_ranking_results():
        """Create mock error ranking results for multiple configs."""
        # Results structure: config_name -> SupportedConfigAnalyzeStep -> condition -> metrics
        return {
            "config1": {
                SupportedConfigAnalyzeStep.ERROR_RANKING: {
                    "main vs. other": {
                        ErrorRankingMetrics.AP: PermutationTestResult(
                            observed_diff=0.1, p_value=0.05,
                            metric_graph1=0.8, metric_graph2=0.7
                        ),
                        ErrorRankingMetrics.TOP_K: {
                            5: PermutationTestResult(
                                observed_diff=0.2, p_value=0.03,
                                metric_graph1=0.6, metric_graph2=0.4
                            ),
                            10: PermutationTestResult(
                                observed_diff=0.15, p_value=0.04,
                                metric_graph1=0.5, metric_graph2=0.35
                            ),
                        },
                        ErrorRankingMetrics.NDCG: {
                            5: PermutationTestResult(
                                observed_diff=0.05, p_value=0.1,
                                metric_graph1=0.9, metric_graph2=0.85
                            ),
                            10: PermutationTestResult(
                                observed_diff=0.03, p_value=0.2,
                                metric_graph1=0.88, metric_graph2=0.85
                            ),
                        }
                    }
                }
            },
            "config2": {
                SupportedConfigAnalyzeStep.ERROR_RANKING: {
                    "main vs. other": {
                        ErrorRankingMetrics.AP: PermutationTestResult(
                            observed_diff=0.15, p_value=0.02,
                            metric_graph1=0.85, metric_graph2=0.7
                        ),
                        ErrorRankingMetrics.TOP_K: {
                            5: PermutationTestResult(
                                observed_diff=0.25, p_value=0.01,
                                metric_graph1=0.65, metric_graph2=0.4
                            ),
                            10: PermutationTestResult(
                                observed_diff=0.2, p_value=0.02,
                                metric_graph1=0.55, metric_graph2=0.35
                            ),
                        },
                        ErrorRankingMetrics.NDCG: {
                            5: PermutationTestResult(
                                observed_diff=0.08, p_value=0.05,
                                metric_graph1=0.92, metric_graph2=0.84
                            ),
                            10: PermutationTestResult(
                                observed_diff=0.05, p_value=0.1,
                                metric_graph1=0.9, metric_graph2=0.85
                            ),
                        }
                    }
                }
            }
        }

    @staticmethod
    def create_replacement_model_results():
        """Create mock replacement model results for multiple configs."""
        return {
            "config1": {
                "memorized": {
                    ReplacementAccuracyMetrics.LAST_TOKEN_COSINE.value: 0.85,
                    ReplacementAccuracyMetrics.KL_DIVERGENCE.value: 0.15,
                    ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value: "hello",
                },
                "random": {
                    ReplacementAccuracyMetrics.LAST_TOKEN_COSINE.value: 0.95,
                    ReplacementAccuracyMetrics.KL_DIVERGENCE.value: 0.05,
                    ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value: "world",
                }
            },
            "config2": {
                "memorized": {
                    ReplacementAccuracyMetrics.LAST_TOKEN_COSINE.value: 0.80,
                    ReplacementAccuracyMetrics.KL_DIVERGENCE.value: 0.20,
                    ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value: "test",
                },
                "random": {
                    ReplacementAccuracyMetrics.LAST_TOKEN_COSINE.value: 0.92,
                    ReplacementAccuracyMetrics.KL_DIVERGENCE.value: 0.08,
                    ReplacementAccuracyMetrics.ORIGINAL_TOP_TOKEN.value: "data",
                }
            }
        }

    @staticmethod
    def create_subgraph_filter_results():
        """Create mock subgraph filter results for multiple configs."""
        return {
            "config1": {
                SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: {
                    DIFF_KEY: {
                        "memorized": {
                            ComparisonMetrics.JACCARD_INDEX: 0.65,
                            ComparisonMetrics.WEIGHTED_JACCARD: 0.72,
                            ComparisonMetrics.OUTPUT_PROBABILITY: 0.85,
                        },
                        "random": {
                            ComparisonMetrics.JACCARD_INDEX: 0.45,
                            ComparisonMetrics.WEIGHTED_JACCARD: 0.52,
                            ComparisonMetrics.OUTPUT_PROBABILITY: 0.60,
                        },
                        SHARED_FEATURES_KEY: {
                            SharedFeatureMetrics.NUM_SHARED: 10,
                            SharedFeatureMetrics.NUM_PROMPTS: 2,
                            SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT: 15.0,
                            SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT: 0.67,
                            "count_at_2pct": 8,
                            "count_at_5pct": 5,
                        }
                    },
                    SIM_KEY: {
                        "memorized": {
                            ComparisonMetrics.JACCARD_INDEX: 0.75,
                            ComparisonMetrics.WEIGHTED_JACCARD: 0.80,
                            ComparisonMetrics.OUTPUT_PROBABILITY: 0.90,
                        },
                        SHARED_FEATURES_KEY: {
                            SharedFeatureMetrics.NUM_SHARED: 12,
                            SharedFeatureMetrics.NUM_PROMPTS: 3,
                            SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT: 18.0,
                            SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT: 0.72,
                            "count_at_2pct": 10,
                            "count_at_5pct": 7,
                        }
                    }
                }
            },
            "config2": {
                SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: {
                    DIFF_KEY: {
                        "memorized": {
                            ComparisonMetrics.JACCARD_INDEX: 0.58,
                            ComparisonMetrics.WEIGHTED_JACCARD: 0.65,
                            ComparisonMetrics.OUTPUT_PROBABILITY: 0.78,
                        },
                        SHARED_FEATURES_KEY: {
                            SharedFeatureMetrics.NUM_SHARED: 8,
                            SharedFeatureMetrics.NUM_PROMPTS: 2,
                            SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT: 12.0,
                            SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT: 0.60,
                            "count_at_2pct": 6,
                            "count_at_5pct": 4,
                        }
                    }
                }
            }
        }

    @staticmethod
    def create_feature_overlap_results():
        """Create mock feature overlap results for multiple configs."""
        return {
            "config1": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: {
                    FeatureSharingMetrics.UNIQUE_FRAC: 0.35,
                    FeatureSharingMetrics.SHARED_FRAC: 0.65,
                    FeatureSharingMetrics.UNIQUE_WEIGHT: 0.40,
                    FeatureSharingMetrics.SHARED_WEIGHT: 0.60,
                }
            },
            "config2": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: {
                    FeatureSharingMetrics.UNIQUE_FRAC: 0.42,
                    FeatureSharingMetrics.SHARED_FRAC: 0.58,
                    FeatureSharingMetrics.UNIQUE_WEIGHT: 0.45,
                    FeatureSharingMetrics.SHARED_WEIGHT: 0.55,
                }
            }
        }


class TestCrossConfigAnalyzerInit(unittest.TestCase):
    """Tests for CrossConfigAnalyzer initialization."""

    def test_init_stores_config_results(self):
        """Test that init stores config results."""
        config_results = {"config1": {}}
        analyzer = CrossConfigAnalyzer(config_results)

        self.assertEqual(analyzer.config_results, config_results)

    def test_init_stores_save_path(self):
        """Test that init stores save path."""
        config_results = {}
        save_path = Path("/tmp/test")

        analyzer = CrossConfigAnalyzer(config_results, save_path=save_path)

        self.assertEqual(analyzer.save_path, save_path)

    def test_init_save_path_defaults_to_none(self):
        """Test that save_path defaults to None."""
        analyzer = CrossConfigAnalyzer({})

        self.assertIsNone(analyzer.save_path)


class TestCrossConfigAnalyzerRun(unittest.TestCase):
    """Tests for CrossConfigAnalyzer.run method."""

    def test_run_iterates_all_steps(self):
        """Test that run iterates through all registered steps."""
        config_results = TestCrossConfigAnalyzerFixtures.create_error_ranking_results()

        # Mock all step classes
        mock_steps = {}
        for step_type in STEP2CLASS:
            mock_step = MagicMock()
            mock_step.run.return_value = {"result": step_type.value}
            mock_steps[step_type] = MagicMock(return_value=mock_step)

        with patch.dict(STEP2CLASS, mock_steps):
            analyzer = CrossConfigAnalyzer(config_results)
            results = analyzer.run()

            # Should have results for all steps
            self.assertEqual(len(results), len(STEP2CLASS))

    def test_run_skips_none_results(self):
        """Test that run skips steps that return None."""
        config_results = {}

        mock_step = MagicMock()
        mock_step.run.return_value = None

        with patch.dict(STEP2CLASS, {SupportedConfigAnalyzeStep.ERROR_RANKING: MagicMock(return_value=mock_step)}):
            analyzer = CrossConfigAnalyzer(config_results)
            results = analyzer.run()

            self.assertNotIn(SupportedConfigAnalyzeStep.ERROR_RANKING, results)

    def test_run_creates_save_path_directory(self):
        """Test that run creates save_path directory if specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "new_dir"
            config_results = {}

            # Mock step to return None so we don't need real step logic
            mock_step = MagicMock()
            mock_step.run.return_value = None

            with patch.dict(STEP2CLASS, {SupportedConfigAnalyzeStep.ERROR_RANKING: MagicMock(return_value=mock_step)}):
                analyzer = CrossConfigAnalyzer(config_results, save_path=save_path)
                analyzer.run()

                self.assertTrue(save_path.exists())


class TestCrossConfigErrorRankingStep(unittest.TestCase):
    """Tests for CrossConfigErrorRankingStep."""

    def test_init_stores_save_path(self):
        """Test that init stores save path."""
        save_path = Path("/tmp/test")
        step = CrossConfigErrorRankingStep(save_path=save_path)

        self.assertEqual(step.save_path, save_path)

    def test_extract_metric_deltas(self):
        """Test extracting delta values from condition results."""
        condition_results = [
            {
                ErrorRankingMetrics.AP: PermutationTestResult(
                    observed_diff=0.1, p_value=0.05,
                    metric_graph1=0.8, metric_graph2=0.7
                ),
                ErrorRankingMetrics.TOP_K: {
                    5: PermutationTestResult(
                        observed_diff=0.2, p_value=0.03,
                        metric_graph1=0.6, metric_graph2=0.4
                    )
                }
            },
            {
                ErrorRankingMetrics.AP: PermutationTestResult(
                    observed_diff=0.15, p_value=0.02,
                    metric_graph1=0.85, metric_graph2=0.7
                ),
                ErrorRankingMetrics.TOP_K: {
                    5: PermutationTestResult(
                        observed_diff=0.25, p_value=0.01,
                        metric_graph1=0.65, metric_graph2=0.4
                    )
                }
            }
        ]

        step = CrossConfigErrorRankingStep()
        deltas = step.extract_metric_deltas(condition_results)

        # Check AP deltas (0.8-0.7=0.1, 0.85-0.7=0.15)
        self.assertEqual(len(deltas[ErrorRankingMetrics.AP]), 2)
        self.assertAlmostEqual(deltas[ErrorRankingMetrics.AP][0], 0.1, places=4)
        self.assertAlmostEqual(deltas[ErrorRankingMetrics.AP][1], 0.15, places=4)

        # Check TOP_K deltas at k=5
        self.assertEqual(len(deltas[ErrorRankingMetrics.TOP_K][5]), 2)
        self.assertAlmostEqual(deltas[ErrorRankingMetrics.TOP_K][5][0], 0.2, places=4)

    def test_extract_raw_values(self):
        """Test extracting raw metric values from condition results."""
        condition_results = [
            {
                ErrorRankingMetrics.AP: PermutationTestResult(
                    observed_diff=0.1, p_value=0.05,
                    metric_graph1=0.8, metric_graph2=0.7
                )
            }
        ]

        step = CrossConfigErrorRankingStep()
        g1_values, g2_values = step.extract_raw_values(condition_results)

        self.assertEqual(g1_values[ErrorRankingMetrics.AP][0], 0.8)
        self.assertEqual(g2_values[ErrorRankingMetrics.AP][0], 0.7)

    def test_condition_level_stats(self):
        """Test computing condition-level statistics from deltas."""
        deltas = {
            ErrorRankingMetrics.AP: [0.1, 0.15, 0.12, 0.08, 0.11],
            ErrorRankingMetrics.TOP_K: {},
            ErrorRankingMetrics.NDCG: {}
        }

        step = CrossConfigErrorRankingStep()
        stats = step.condition_level_stats(deltas)

        # Should have stats for AP
        self.assertIn(ErrorRankingMetrics.AP, stats)
        ap_stats = stats[ErrorRankingMetrics.AP]

        # Check mean and median
        self.assertAlmostEqual(ap_stats["mean"], np.mean([0.1, 0.15, 0.12, 0.08, 0.11]), places=4)
        self.assertAlmostEqual(ap_stats["median"], np.median([0.1, 0.15, 0.12, 0.08, 0.11]), places=4)

        # Check count fields
        self.assertEqual(ap_stats["count_g1_gt_g2"], 5)  # All positive
        self.assertEqual(ap_stats["count_g1_lt_g2"], 0)
        self.assertEqual(ap_stats["count_equal"], 0)

    def test_pooled_condition_stats(self):
        """Test computing pooled statistics across conditions."""
        all_deltas = {
            "condition1": {
                ErrorRankingMetrics.AP: [0.1, 0.15],
                ErrorRankingMetrics.TOP_K: {},
                ErrorRankingMetrics.NDCG: {}
            },
            "condition2": {
                ErrorRankingMetrics.AP: [0.12, 0.08],
                ErrorRankingMetrics.TOP_K: {},
                ErrorRankingMetrics.NDCG: {}
            }
        }

        step = CrossConfigErrorRankingStep()
        stats, pooled_deltas = step.pooled_condition_stats(all_deltas)

        # Pooled deltas should combine both conditions
        self.assertEqual(len(pooled_deltas[ErrorRankingMetrics.AP]), 4)

        # Stats should reflect pooled data
        self.assertIn(ErrorRankingMetrics.AP, stats)
        self.assertEqual(stats[ErrorRankingMetrics.AP]["n_samples"], 4)

    def test_init_metric_structure(self):
        """Test initializing empty metric structure."""
        step = CrossConfigErrorRankingStep()
        structure = step._init_metric_structure()

        # K_METRICS should have empty dicts
        self.assertIsInstance(structure[ErrorRankingMetrics.TOP_K], dict)
        self.assertIsInstance(structure[ErrorRankingMetrics.NDCG], dict)

        # Non-K metrics should have empty lists
        self.assertIsInstance(structure[ErrorRankingMetrics.AP], list)

    def test_run_returns_none_when_no_error_ranking_results(self):
        """Test that run returns None when no configs have error ranking results."""
        config_results = {
            "config1": {
                SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: {}  # Different step
            }
        }

        step = CrossConfigErrorRankingStep()
        result = step.run(config_results)

        self.assertIsNone(result)


class TestCrossConfigReplacementModelAccuracyStep(unittest.TestCase):
    """Tests for CrossConfigReplacementModelAccuracyStep."""

    def test_is_significant(self):
        """Test significance threshold checking."""
        self.assertTrue(CrossConfigReplacementModelAccuracyStep.is_significant(0.01))
        self.assertTrue(CrossConfigReplacementModelAccuracyStep.is_significant(0.049))
        self.assertFalse(CrossConfigReplacementModelAccuracyStep.is_significant(0.05))
        self.assertFalse(CrossConfigReplacementModelAccuracyStep.is_significant(0.1))

    def test_is_significant_array(self):
        """Test significance checking with arrays."""
        p_values = np.array([0.01, 0.05, 0.1])

        result = CrossConfigReplacementModelAccuracyStep.is_significant(p_values)

        np.testing.assert_array_equal(result, [True, False, False])

    def test_compute_significance_stats(self):
        """Test computing significance statistics between groups."""
        from src.metrics import SignificanceMetrics

        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        step = CrossConfigReplacementModelAccuracyStep()
        stats = step.compute_significance_stats(group1, group2)

        self.assertIsInstance(stats, dict)
        self.assertAlmostEqual(stats[SignificanceMetrics.GROUP1_MEAN.value], 3.0, places=4)
        self.assertAlmostEqual(stats[SignificanceMetrics.GROUP2_MEAN.value], 8.0, places=4)
        self.assertEqual(stats[SignificanceMetrics.N_PER_GROUP.value], 5)

        # Groups are clearly different, so should be significant
        self.assertLess(stats[SignificanceMetrics.T_P_VALUE.value], 0.05)

    def test_apply_bh_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        df = pd.DataFrame({
            "metric": ["m1", "m2", "m3"],
            "t_p_value": [0.01, 0.03, 0.04],
            "mw_p_value": [0.02, 0.04, 0.06]
        })

        step = CrossConfigReplacementModelAccuracyStep()
        corrected_df = step.apply_bh_correction(df)

        # Should add BH-corrected columns
        self.assertIn("t_p_value_bh", corrected_df.columns)
        self.assertIn("mw_p_value_bh", corrected_df.columns)
        self.assertIn("t_significant_bh", corrected_df.columns)
        self.assertIn("mw_significant_bh", corrected_df.columns)

    def test_get_token_complexity(self):
        """Test token complexity calculation."""
        complexity = CrossConfigReplacementModelAccuracyStep.get_token_complexity("hello")

        self.assertIn(ComplexityMetrics.ZIPF_FREQUENCY, complexity)
        self.assertIn(ComplexityMetrics.TOKEN_LENGTH, complexity)
        self.assertEqual(complexity[ComplexityMetrics.TOKEN_LENGTH], 5)
        # "hello" is a common word, should have high Zipf frequency
        self.assertGreater(complexity[ComplexityMetrics.ZIPF_FREQUENCY], 0)

    def test_get_token_complexity_with_space(self):
        """Test token complexity with leading space (tokenizer artifact)."""
        complexity = CrossConfigReplacementModelAccuracyStep.get_token_complexity(" hello")

        # Should strip the space
        self.assertEqual(complexity[ComplexityMetrics.TOKEN_LENGTH], 5)

    def test_get_token_complexity_empty(self):
        """Test token complexity for empty/whitespace token."""
        complexity = CrossConfigReplacementModelAccuracyStep.get_token_complexity("   ")

        self.assertEqual(complexity[ComplexityMetrics.TOKEN_LENGTH], 0)
        self.assertEqual(complexity[ComplexityMetrics.ZIPF_FREQUENCY], 0.0)

    def test_to_df(self):
        """Test converting results dict to DataFrame."""
        results = {
            "config1": {
                "memorized": {"metric1": 0.5, "metric2": 0.8},
                "random": {"metric1": 0.7, "metric2": 0.9}
            },
            "config2": {
                "memorized": {"metric1": 0.4, "metric2": 0.75},
                "random": {"metric1": 0.65, "metric2": 0.85}
            }
        }

        step = CrossConfigReplacementModelAccuracyStep()
        df = step._to_df(results)

        self.assertEqual(len(df), 4)  # 2 configs * 2 conditions
        self.assertIn("condition", df.columns)
        self.assertIn("config", df.columns)
        self.assertIn("metric1", df.columns)
        self.assertIn("metric2", df.columns)

    def test_df_to_results_dict(self):
        """Test converting DataFrame to results dictionary."""
        df = pd.DataFrame({
            "metric": ["metric1", "metric2"],
            "group1_mean": [0.5, 0.6],
            "group2_mean": [0.7, 0.8],
            "group1_std": [0.1, 0.15],
            "group2_std": [0.12, 0.18],
            "n_per_group": [10, 10],
            "t_statistic": [-2.5, -1.8],
            "t_p_value": [0.02, 0.08],
            "t_significant": [True, False],
            "mann_whitney_u": [30, 40],
            "mw_p_value": [0.03, 0.1],
            "mw_significant": [True, False],
            "cohens_d": [-0.8, -0.5],
            "rank_biserial_r": [-0.4, -0.2]
        })

        step = CrossConfigReplacementModelAccuracyStep()
        results = step.df_to_results_dict(df)

        self.assertIn("metric1", results)
        self.assertIn("metric2", results)
        self.assertEqual(results["metric1"]["group1_mean"], 0.5)
        self.assertEqual(results["metric2"]["t_significant"], False)


class TestCrossConfigSubgraphFilterStep(unittest.TestCase):
    """Tests for CrossConfigSubgraphFilterStep."""

    def test_init_stores_save_path(self):
        """Test that init stores save path."""
        save_path = Path("/tmp/test")
        step = CrossConfigSubgraphFilterStep(save_path=save_path)

        self.assertEqual(step.save_path, save_path)

    def test_init_uses_default_thresholds(self):
        """Test that init uses default thresholds when not provided."""
        from src.graph_analyzer import GraphAnalyzer

        step = CrossConfigSubgraphFilterStep()

        self.assertEqual(step.thresholds, GraphAnalyzer.DEFAULT_THRESHOLDS)

    def test_init_uses_custom_thresholds(self):
        """Test that init accepts custom thresholds."""
        thresholds = [1, 3, 5]
        step = CrossConfigSubgraphFilterStep(thresholds=thresholds)

        self.assertEqual(step.thresholds, thresholds)

    def test_intersection_metric_cols_returns_comparison_metrics(self):
        """Test that intersection_metric_cols returns all ComparisonMetrics values."""
        step = CrossConfigSubgraphFilterStep()
        cols = step.intersection_metric_cols

        for metric in ComparisonMetrics:
            self.assertIn(metric.value, cols)

    def test_shared_feature_metric_cols_includes_base_cols(self):
        """Test that shared_feature_metric_cols includes base metrics."""
        step = CrossConfigSubgraphFilterStep()
        cols = step.shared_feature_metric_cols

        self.assertIn(SharedFeatureMetrics.NUM_SHARED.value, cols)
        self.assertIn(SharedFeatureMetrics.NUM_PROMPTS.value, cols)
        self.assertIn(SharedFeatureMetrics.AVG_FEATURES_PER_PROMPT.value, cols)
        self.assertIn(SharedFeatureMetrics.SHARED_PRESENT_PER_PROMPT.value, cols)

    def test_shared_feature_metric_cols_includes_threshold_cols(self):
        """Test that shared_feature_metric_cols includes threshold-based columns."""
        thresholds = [2, 5, 10]
        step = CrossConfigSubgraphFilterStep(thresholds=thresholds)
        cols = step.shared_feature_metric_cols

        for t in thresholds:
            expected_col = SharedFeatureMetrics.COUNT_AT_THRESHOLD.value.format(t)
            self.assertIn(expected_col, cols)

    def test_run_returns_none_when_no_subgraph_filter_results(self):
        """Test that run returns None when no configs have subgraph filter results."""
        config_results = {
            "config1": {
                SupportedConfigAnalyzeStep.ERROR_RANKING: {}  # Different step
            }
        }

        step = CrossConfigSubgraphFilterStep()
        result = step.run(config_results)

        self.assertIsNone(result)

    def test_run_extracts_intersection_metrics(self):
        """Test that run extracts intersection metrics correctly."""
        config_results = TestCrossConfigAnalyzerFixtures.create_subgraph_filter_results()

        step = CrossConfigSubgraphFilterStep(thresholds=[2, 5])
        result = step.run(config_results)

        self.assertIsNotNone(result)
        self.assertIn("intersection_metrics", result)

        df = result["intersection_metrics"]
        self.assertIn(CONFIG_NAME_COL, df.columns)
        self.assertIn(PROMPT_TYPE_COL, df.columns)

        # Check metric columns exist
        for metric in ComparisonMetrics:
            self.assertIn(metric.value, df.columns)

    def test_run_extracts_shared_feature_metrics(self):
        """Test that run extracts shared feature metrics correctly."""
        config_results = TestCrossConfigAnalyzerFixtures.create_subgraph_filter_results()

        step = CrossConfigSubgraphFilterStep(thresholds=[2, 5])
        result = step.run(config_results)

        self.assertIsNotNone(result)
        self.assertIn(SHARED_FEATURES_KEY, result)

        df = result[SHARED_FEATURES_KEY]
        self.assertIn(CONFIG_NAME_COL, df.columns)

    def test_run_skips_shared_features_key_in_intersection(self):
        """Test that shared_features key is not included as prompt type."""
        config_results = TestCrossConfigAnalyzerFixtures.create_subgraph_filter_results()

        step = CrossConfigSubgraphFilterStep(thresholds=[2, 5])
        result = step.run(config_results)

        df = result["intersection_metrics"]
        prompt_types = df[PROMPT_TYPE_COL].unique()

        self.assertNotIn(SHARED_FEATURES_KEY, prompt_types)

    def test_run_saves_to_csv_when_save_path_provided(self):
        """Test that run saves results to CSV when save_path is provided."""
        config_results = TestCrossConfigAnalyzerFixtures.create_subgraph_filter_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            step = CrossConfigSubgraphFilterStep(save_path=save_path, thresholds=[2, 5])
            step.run(config_results)

            # Check CSVs were created
            self.assertTrue((save_path / CrossConfigSubgraphFilterStep.OVERLAP_ANALYSIS_FILENAME).exists())
            self.assertTrue((save_path / CrossConfigSubgraphFilterStep.SHARED_FEATURE_METRICS_FILENAME).exists())

    def test_init_intersection_results_structure(self):
        """Test that _init_intersection_results creates correct structure."""
        step = CrossConfigSubgraphFilterStep()
        results = step._init_intersection_results()

        self.assertIn(CONFIG_NAME_COL, results)
        self.assertIn(PROMPT_TYPE_COL, results)
        for metric in ComparisonMetrics:
            self.assertIn(metric.value, results)

    def test_init_shared_feature_results_structure(self):
        """Test that _init_shared_feature_results creates correct structure."""
        step = CrossConfigSubgraphFilterStep(thresholds=[2, 5])
        results = step._init_shared_feature_results()

        self.assertIn(CONFIG_NAME_COL, results)
        self.assertIn(SharedFeatureMetrics.NUM_SHARED.value, results)
        self.assertIn("count_at_2pct", results)
        self.assertIn("count_at_5pct", results)


class TestCrossConfigFeatureOverlapStep(unittest.TestCase):
    """Tests for CrossConfigFeatureOverlapStep."""

    def test_init_stores_save_path(self):
        """Test that init stores save path."""
        save_path = Path("/tmp/test")
        step = CrossConfigFeatureOverlapStep(save_path=save_path)

        self.assertEqual(step.save_path, save_path)

    def test_metric_cols_returns_feature_sharing_metrics(self):
        """Test that metric_cols returns all FeatureSharingMetrics values."""
        step = CrossConfigFeatureOverlapStep()
        cols = step.metric_cols

        for metric in FeatureSharingMetrics:
            self.assertIn(metric.value, cols)

    def test_run_returns_none_when_no_feature_overlap_results(self):
        """Test that run returns None when no configs have feature overlap results."""
        config_results = {
            "config1": {
                SupportedConfigAnalyzeStep.ERROR_RANKING: {}  # Different step
            }
        }

        step = CrossConfigFeatureOverlapStep()
        result = step.run(config_results)

        self.assertIsNone(result)

    def test_run_returns_dataframe_with_metrics(self):
        """Test that run returns DataFrame with feature sharing metrics."""
        config_results = TestCrossConfigAnalyzerFixtures.create_feature_overlap_results()

        step = CrossConfigFeatureOverlapStep()
        result = step.run(config_results)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(FEATURE_CONFIG_NAME_COL, result.columns)

        for metric in FeatureSharingMetrics:
            self.assertIn(metric.value, result.columns)

    def test_run_includes_all_configs(self):
        """Test that run includes results from all configs."""
        config_results = TestCrossConfigAnalyzerFixtures.create_feature_overlap_results()

        step = CrossConfigFeatureOverlapStep()
        result = step.run(config_results)

        config_names = result[FEATURE_CONFIG_NAME_COL].tolist()
        self.assertIn("config1", config_names)
        self.assertIn("config2", config_names)

    def test_run_extracts_correct_values(self):
        """Test that run extracts correct metric values."""
        config_results = TestCrossConfigAnalyzerFixtures.create_feature_overlap_results()

        step = CrossConfigFeatureOverlapStep()
        result = step.run(config_results)

        config1_row = result[result[FEATURE_CONFIG_NAME_COL] == "config1"].iloc[0]
        self.assertAlmostEqual(config1_row[FeatureSharingMetrics.UNIQUE_FRAC.value], 0.35, places=2)
        self.assertAlmostEqual(config1_row[FeatureSharingMetrics.SHARED_FRAC.value], 0.65, places=2)

    def test_run_saves_to_csv_when_save_path_provided(self):
        """Test that run saves results to CSV when save_path is provided."""
        config_results = TestCrossConfigAnalyzerFixtures.create_feature_overlap_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            step = CrossConfigFeatureOverlapStep(save_path=save_path)
            step.run(config_results)

            csv_path = save_path / CrossConfigFeatureOverlapStep.FEATURE_OVERLAP_METRICS_FILENAME
            self.assertTrue(csv_path.exists())

            # Verify CSV content
            saved_df = pd.read_csv(csv_path, index_col=0)
            self.assertEqual(len(saved_df), 2)  # Two configs

    def test_init_results_creates_empty_structure(self):
        """Test that _init_results creates correct empty structure."""
        step = CrossConfigFeatureOverlapStep()
        results = step._init_results()

        self.assertIn(FEATURE_CONFIG_NAME_COL, results)
        self.assertEqual(results[FEATURE_CONFIG_NAME_COL], [])

        for metric in FeatureSharingMetrics:
            self.assertIn(metric.value, results)
            self.assertEqual(results[metric.value], [])


class TestStepMapping(unittest.TestCase):
    """Tests for STEP2CLASS mapping."""

    def test_expected_steps_are_mapped(self):
        """Test that expected cross-config steps are mapped."""
        expected_steps = [
            SupportedConfigAnalyzeStep.ERROR_RANKING,
            SupportedConfigAnalyzeStep.REPLACEMENT_MODEL,
            SupportedConfigAnalyzeStep.SUBGRAPH_FILTER,
            SupportedConfigAnalyzeStep.FEATURE_OVERLAP,
        ]

        for step in expected_steps:
            self.assertIn(step, STEP2CLASS)

    def test_mapped_steps_are_cross_config_steps(self):
        """Test that all mapped classes are CrossConfigAnalyzeStep subclasses."""
        from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep

        for step_type, step_cls in STEP2CLASS.items():
            self.assertTrue(
                issubclass(step_cls, CrossConfigAnalyzeStep),
                f"{step_cls} is not a subclass of CrossConfigAnalyzeStep"
            )


class TestCrossConfigL0ReplacementModelStep(unittest.TestCase):
    """Tests for CrossConfigL0ReplacementModelStep."""

    def test_init_stores_paths(self):
        """Test that init stores save_path and load_path."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            CrossConfigL0ReplacementModelStep
        )

        save_path = Path("/tmp/save")
        load_path = Path("/tmp/load")
        step = CrossConfigL0ReplacementModelStep(save_path=save_path, load_path=load_path)

        self.assertEqual(step.save_path, save_path)
        self.assertEqual(step.load_path, load_path)

    def test_init_load_path_defaults_to_save_path(self):
        """Test that load_path defaults to save_path when not provided."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            CrossConfigL0ReplacementModelStep
        )

        save_path = Path("/tmp/save")
        step = CrossConfigL0ReplacementModelStep(save_path=save_path)

        self.assertEqual(step.load_path, save_path)

    def test_extract_results_handles_new_format(self):
        """Test _extract_results handles new format with d_transcoder."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            CrossConfigL0ReplacementModelStep
        )

        config_results = {
            "config1": {
                SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: {
                    "results": {"prompt1": [1.0, 2.0, 3.0]},
                    "d_transcoder": 1000
                }
            }
        }

        step = CrossConfigL0ReplacementModelStep(save_path=Path("/tmp"))
        results = step._extract_results(config_results)

        self.assertIsNotNone(results)
        self.assertIn("config1", results)
        self.assertEqual(results["config1"]["d_transcoder"], 1000)

    def test_compute_layer_stats(self):
        """Test _compute_layer_stats returns correct statistics."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            CrossConfigL0ReplacementModelStep, L0_VALUE_COL, LAYER_COL
        )

        df = pd.DataFrame({
            LAYER_COL: [0, 0, 0, 1, 1, 1],
            L0_VALUE_COL: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })

        step = CrossConfigL0ReplacementModelStep(save_path=Path("/tmp"))
        stats = step._compute_layer_stats(df)

        self.assertEqual(len(stats), 2)  # Two layers
        self.assertAlmostEqual(stats[stats[LAYER_COL] == 0]["mean"].iloc[0], 2.0, places=4)
        self.assertAlmostEqual(stats[stats[LAYER_COL] == 1]["mean"].iloc[0], 5.0, places=4)

    def test_run_returns_none_when_save_path_is_none(self):
        """Test that run returns None when save_path is None."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            CrossConfigL0ReplacementModelStep
        )

        step = CrossConfigL0ReplacementModelStep(save_path=None)
        result = step.run({})

        self.assertIsNone(result)


class TestCrossConfigEarlyLayerContributionStep(unittest.TestCase):
    """Tests for CrossConfigEarlyLayerContributionStep."""

    def test_init_stores_save_path(self):
        """Test that init stores save_path."""
        from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
            CrossConfigEarlyLayerContributionStep
        )

        save_path = Path("/tmp/test")
        step = CrossConfigEarlyLayerContributionStep(save_path=save_path)

        self.assertEqual(step.save_path, save_path)

    def test_config_results_key_is_early_layer_contribution(self):
        """Test that step uses EARLY_LAYER_CONTRIBUTION key."""
        from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
            CrossConfigEarlyLayerContributionStep
        )

        step = CrossConfigEarlyLayerContributionStep()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION)

    def test_metric_cols_includes_early_layer_metrics(self):
        """Test that metric_cols includes early layer metrics."""
        from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
            CrossConfigEarlyLayerContributionStep
        )
        from src.metrics import EarlyLayerMetrics

        step = CrossConfigEarlyLayerContributionStep()
        cols = step.metric_cols

        for metric in EarlyLayerMetrics:
            self.assertIn(metric.value, cols)

    def test_run_returns_none_when_no_results(self):
        """Test that run returns None when no early layer results found."""
        from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
            CrossConfigEarlyLayerContributionStep
        )

        config_results = {"config1": {}}
        step = CrossConfigEarlyLayerContributionStep()
        result = step.run(config_results)

        self.assertIsNone(result)

    def test_run_returns_dataframe(self):
        """Test that run returns DataFrame with correct columns."""
        from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
            CrossConfigEarlyLayerContributionStep, EARLY_LAYER_FRACTION_COL, MAX_LAYER_COL
        )
        from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
            CONFIG_NAME_COL, PROMPT_TYPE_COL
        )

        config_results = {
            "config1": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION: {
                    "memorized": {2: 0.5, 3: 0.6},
                    "random": {2: 0.3, 3: 0.4}
                }
            }
        }

        step = CrossConfigEarlyLayerContributionStep()
        result = step.run(config_results)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(CONFIG_NAME_COL, result.columns)
        self.assertIn(PROMPT_TYPE_COL, result.columns)
        self.assertIn(MAX_LAYER_COL, result.columns)
        self.assertIn(EARLY_LAYER_FRACTION_COL, result.columns)
        self.assertEqual(len(result), 4)  # 2 prompt types * 2 max_layers


if __name__ == "__main__":
    unittest.main()
