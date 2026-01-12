import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import pandas as pd

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_analyzer import (
    CrossConditionAnalyzer, CROSS_CONDITION_STEPS
)
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import CrossConditionAnalyzeStep
from src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step import (
    CrossConditionOverlapVisualizationStep, INTERSECTION_METRICS_KEY
)
from src.analysis.cross_condition_analysis.cross_condition_feature_overlap_visualization_step import (
    CrossConditionFeatureOverlapVisualizationStep
)
from src.analysis.cross_condition_analysis.cross_condition_shared_features_visualization_step import (
    CrossConditionSharedFeaturesVisualizationStep
)
from src.analysis.cross_condition_analysis.cross_condition_early_layer_step import (
    CrossConditionEarlyLayerStep, DEFAULT_PRIMARY_THRESHOLD
)
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import (
    CONFIG_NAME_COL, PROMPT_TYPE_COL, SHARED_FEATURES_KEY
)
from src.analysis.cross_config_analysis.cross_config_early_layer_contribution_step import (
    EARLY_LAYER_FRACTION_COL, MAX_LAYER_COL
)
from src.analysis.cross_condition_analysis.cross_condition_analyze_step import DEFAULT_CONDITION_COL as CONDITION_COL
from src.metrics import ComparisonMetrics, FeatureSharingMetrics


class TestCrossConditionAnalyzerFixtures:
    """Shared test fixtures for cross-condition analyzer tests."""

    @staticmethod
    def create_intersection_metrics_df() -> pd.DataFrame:
        """Create mock intersection metrics DataFrame."""
        return pd.DataFrame({
            CONFIG_NAME_COL: ["config1", "config1", "config2", "config2"],
            PROMPT_TYPE_COL: ["memorized", "random", "memorized", "random"],
            ComparisonMetrics.JACCARD_INDEX.value: [0.65, 0.45, 0.58, 0.42],
            ComparisonMetrics.WEIGHTED_JACCARD.value: [0.72, 0.52, 0.65, 0.48],
            ComparisonMetrics.OUTPUT_PROBABILITY.value: [0.85, 0.60, 0.78, 0.55],
        })

    @staticmethod
    def create_shared_features_df() -> pd.DataFrame:
        """Create mock shared features DataFrame."""
        return pd.DataFrame({
            CONFIG_NAME_COL: ["config1", "config2"],
            "num_shared": [10, 8],
            "num_prompts": [2, 2],
            "avg_features_per_prompt": [15.0, 12.0],
            "shared_present_per_prompt": ["8,10", "6,8"],
            "count_at_50pct": [8, 6],
            "count_at_75pct": [5, 4],
            "count_at_100pct": [2, 1],
        })

    @staticmethod
    def create_feature_overlap_df() -> pd.DataFrame:
        """Create mock feature overlap DataFrame."""
        return pd.DataFrame({
            CONFIG_NAME_COL: ["config1", "config2"],
            FeatureSharingMetrics.UNIQUE_FRAC.value: [0.35, 0.42],
            FeatureSharingMetrics.SHARED_FRAC.value: [0.65, 0.58],
            FeatureSharingMetrics.UNIQUE_WEIGHT.value: [0.40, 0.45],
            FeatureSharingMetrics.SHARED_WEIGHT.value: [0.60, 0.55],
        })

    @staticmethod
    def create_early_layer_df(max_layers: list = None) -> pd.DataFrame:
        """Create mock early layer contribution DataFrame."""
        if max_layers is None:
            max_layers = [2]

        rows = []
        for max_layer in max_layers:
            rows.extend([
                {CONFIG_NAME_COL: "config1", PROMPT_TYPE_COL: "memorized",
                 MAX_LAYER_COL: max_layer, EARLY_LAYER_FRACTION_COL: 0.5 + max_layer * 0.05},
                {CONFIG_NAME_COL: "config1", PROMPT_TYPE_COL: "random",
                 MAX_LAYER_COL: max_layer, EARLY_LAYER_FRACTION_COL: 0.3 + max_layer * 0.05},
                {CONFIG_NAME_COL: "config2", PROMPT_TYPE_COL: "memorized",
                 MAX_LAYER_COL: max_layer, EARLY_LAYER_FRACTION_COL: 0.55 + max_layer * 0.05},
                {CONFIG_NAME_COL: "config2", PROMPT_TYPE_COL: "random",
                 MAX_LAYER_COL: max_layer, EARLY_LAYER_FRACTION_COL: 0.35 + max_layer * 0.05},
            ])
        return pd.DataFrame(rows)

    @staticmethod
    def create_condition_results() -> Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]:
        """Create mock condition results with all data types."""
        fixtures = TestCrossConditionAnalyzerFixtures

        return {
            "condition1": {
                SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: {
                    INTERSECTION_METRICS_KEY: fixtures.create_intersection_metrics_df(),
                    SHARED_FEATURES_KEY: fixtures.create_shared_features_df(),
                },
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: fixtures.create_feature_overlap_df(),
            },
            "condition2": {
                SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: {
                    INTERSECTION_METRICS_KEY: fixtures.create_intersection_metrics_df(),
                    SHARED_FEATURES_KEY: fixtures.create_shared_features_df(),
                },
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: fixtures.create_feature_overlap_df(),
            },
        }


class TestCrossConditionAnalyzeStep(unittest.TestCase):
    """Tests for CrossConditionAnalyzeStep base class."""

    def test_combine_condition_dataframes_returns_none_when_no_key(self):
        """Test that combine_condition_dataframes returns None when CONFIG_RESULTS_KEY is None."""
        step = CrossConditionOverlapVisualizationStep()
        step.CONFIG_RESULTS_KEY = None

        result = step.combine_condition_dataframes({})

        self.assertIsNone(result)

    def test_combine_condition_dataframes_adds_condition_column(self):
        """Test that combine_condition_dataframes adds condition name as condition column."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: pd.DataFrame({
                    CONFIG_NAME_COL: ["cfg1"],
                    "value": [1.0]
                })
            },
            "cond2": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: pd.DataFrame({
                    CONFIG_NAME_COL: ["cfg1"],
                    "value": [2.0]
                })
            },
        }

        step = CrossConditionFeatureOverlapVisualizationStep()
        result = step.combine_condition_dataframes(condition_results)

        self.assertIn(CONDITION_COL, result.columns)
        self.assertEqual(set(result[CONDITION_COL].unique()), {"cond1", "cond2"})

    def test_combine_condition_dataframes_normalizes_config_names(self):
        """Test that combine_condition_dataframes strips directory prefixes."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: pd.DataFrame({
                    CONFIG_NAME_COL: ["path/to/config1", "another/config2"],
                    "value": [1.0, 2.0]
                })
            },
        }

        step = CrossConditionFeatureOverlapVisualizationStep()
        result = step.combine_condition_dataframes(condition_results)

        self.assertEqual(list(result[CONFIG_NAME_COL]), ["config1", "config2"])

    def test_get_ordering_derives_from_data(self):
        """Test that get_ordering derives order from DataFrame when not provided."""
        df = pd.DataFrame({
            CONDITION_COL: ["b", "a", "c"],
            CONFIG_NAME_COL: ["cfg2", "cfg1", "cfg3"]
        })

        step = CrossConditionOverlapVisualizationStep()
        condition_order, config_order = step.get_ordering(df)

        self.assertEqual(condition_order, ["a", "b", "c"])
        self.assertEqual(config_order, ["cfg1", "cfg2", "cfg3"])

    def test_get_ordering_uses_provided_values(self):
        """Test that get_ordering uses provided values over derived ones."""
        df = pd.DataFrame({
            CONDITION_COL: ["b", "a"],
            CONFIG_NAME_COL: ["cfg2", "cfg1"]
        })

        step = CrossConditionOverlapVisualizationStep(
            condition_order=["x", "y"],
            config_order=["z"]
        )
        condition_order, config_order = step.get_ordering(df)

        self.assertEqual(condition_order, ["x", "y"])
        self.assertEqual(config_order, ["z"])


class TestCrossConditionOverlapVisualizationStep(unittest.TestCase):
    """Tests for CrossConditionOverlapVisualizationStep."""

    def test_config_results_key_is_subgraph_filter(self):
        """Test that step uses SUBGRAPH_FILTER key."""
        step = CrossConditionOverlapVisualizationStep()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.SUBGRAPH_FILTER)

    def test_results_sub_key_is_intersection_metrics(self):
        """Test that step uses intersection_metrics sub-key."""
        step = CrossConditionOverlapVisualizationStep()
        self.assertEqual(step.RESULTS_SUB_KEY, INTERSECTION_METRICS_KEY)

    def test_run_returns_none_when_no_data(self):
        """Test that run returns None when no intersection metrics found."""
        condition_results = {"cond1": {}}

        step = CrossConditionOverlapVisualizationStep()
        result = step.run(condition_results)

        self.assertIsNone(result)

    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_by_condition')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_heatmap')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_line')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_scatter')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_combined')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_correlation_heatmap')
    def test_run_returns_combined_dataframe(self, *mocks):
        """Test that run returns combined DataFrame."""
        condition_results = TestCrossConditionAnalyzerFixtures.create_condition_results()

        step = CrossConditionOverlapVisualizationStep()
        result = step.run(condition_results)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(ComparisonMetrics.JACCARD_INDEX.value, result.columns)


class TestCrossConditionFeatureOverlapVisualizationStep(unittest.TestCase):
    """Tests for CrossConditionFeatureOverlapVisualizationStep."""

    def test_config_results_key_is_feature_overlap(self):
        """Test that step uses FEATURE_OVERLAP key."""
        step = CrossConditionFeatureOverlapVisualizationStep()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.FEATURE_OVERLAP)

    def test_results_sub_key_is_none(self):
        """Test that step has no sub-key (direct DataFrame)."""
        step = CrossConditionFeatureOverlapVisualizationStep()
        self.assertIsNone(step.RESULTS_SUB_KEY)

    def test_run_returns_none_when_no_data(self):
        """Test that run returns None when no feature overlap data found."""
        condition_results = {"cond1": {}}

        step = CrossConditionFeatureOverlapVisualizationStep()
        result = step.run(condition_results)

        self.assertIsNone(result)

    @patch('src.analysis.cross_condition_analysis.cross_condition_feature_overlap_visualization_step.plot_combined_metrics')
    def test_run_adds_condition_column(self, mock_plot):
        """Test that run adds condition column from condition names."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.FEATURE_OVERLAP: TestCrossConditionAnalyzerFixtures.create_feature_overlap_df()
            },
        }

        step = CrossConditionFeatureOverlapVisualizationStep()
        result = step.run(condition_results)

        self.assertIn(step.CONDITION_COL, result.columns)
        self.assertEqual(result[step.CONDITION_COL].unique()[0], "cond1")


class TestCrossConditionSharedFeaturesVisualizationStep(unittest.TestCase):
    """Tests for CrossConditionSharedFeaturesVisualizationStep."""

    def test_config_results_key_is_subgraph_filter(self):
        """Test that step uses SUBGRAPH_FILTER key."""
        step = CrossConditionSharedFeaturesVisualizationStep()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.SUBGRAPH_FILTER)

    def test_results_sub_key_is_shared_features(self):
        """Test that step uses shared_features sub-key."""
        step = CrossConditionSharedFeaturesVisualizationStep()
        self.assertEqual(step.RESULTS_SUB_KEY, SHARED_FEATURES_KEY)

    def test_run_returns_none_when_no_data(self):
        """Test that run returns None when no shared features data found."""
        condition_results = {"cond1": {}}

        step = CrossConditionSharedFeaturesVisualizationStep()
        result = step.run(condition_results)

        self.assertIsNone(result)


class TestCrossConditionAnalyzer(unittest.TestCase):
    """Tests for CrossConditionAnalyzer."""

    def test_init_stores_condition_results(self):
        """Test that init stores condition results."""
        condition_results = {"cond1": {}}
        analyzer = CrossConditionAnalyzer(condition_results)

        self.assertEqual(analyzer.condition_results, condition_results)

    def test_init_stores_save_path(self):
        """Test that init stores save path."""
        save_path = Path("/tmp/test")
        analyzer = CrossConditionAnalyzer({}, save_path=save_path)

        self.assertEqual(analyzer.save_path, save_path)

    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_by_condition')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_heatmap')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_line')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_scatter')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_combined')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_correlation_heatmap')
    @patch('src.analysis.cross_condition_analysis.cross_condition_feature_overlap_visualization_step.plot_combined_metrics')
    @patch('src.analysis.cross_condition_analysis.cross_condition_shared_features_visualization_step.plot_shared_feature_metrics')
    def test_run_returns_dict_of_results(self, *mocks):
        """Test that run returns dictionary of step results."""
        condition_results = TestCrossConditionAnalyzerFixtures.create_condition_results()

        analyzer = CrossConditionAnalyzer(condition_results)
        results = analyzer.run()

        self.assertIsInstance(results, dict)

    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_by_condition')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_heatmap')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_line')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_scatter')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_metric_vs_probability_combined')
    @patch('src.analysis.cross_condition_analysis.cross_condition_overlap_visualization_step.plot_correlation_heatmap')
    @patch('src.analysis.cross_condition_analysis.cross_condition_feature_overlap_visualization_step.plot_combined_metrics')
    @patch('src.analysis.cross_condition_analysis.cross_condition_shared_features_visualization_step.plot_shared_feature_metrics')
    def test_run_includes_step_results_by_class_name(self, *mocks):
        """Test that run returns results keyed by step class name."""
        condition_results = TestCrossConditionAnalyzerFixtures.create_condition_results()

        analyzer = CrossConditionAnalyzer(condition_results)
        results = analyzer.run()

        # Should have results from overlap step at minimum
        self.assertIn("CrossConditionOverlapVisualizationStep", results)

    def test_cross_condition_steps_list_contains_expected_steps(self):
        """Test that CROSS_CONDITION_STEPS contains expected step classes."""
        self.assertIn(CrossConditionOverlapVisualizationStep, CROSS_CONDITION_STEPS)
        self.assertIn(CrossConditionFeatureOverlapVisualizationStep, CROSS_CONDITION_STEPS)
        self.assertIn(CrossConditionSharedFeaturesVisualizationStep, CROSS_CONDITION_STEPS)


class TestCrossConditionEarlyLayerStep(unittest.TestCase):
    """Tests for CrossConditionEarlyLayerStep."""

    def test_config_results_key_is_early_layer_contribution(self):
        """Test that step uses EARLY_LAYER_CONTRIBUTION key."""
        step = CrossConditionEarlyLayerStep()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION)

    def test_default_primary_threshold_is_2(self):
        """Test that default primary_threshold is 2."""
        step = CrossConditionEarlyLayerStep()
        self.assertEqual(step.primary_threshold, DEFAULT_PRIMARY_THRESHOLD)
        self.assertEqual(step.primary_threshold, 2)

    def test_custom_primary_threshold(self):
        """Test that custom primary_threshold is stored."""
        step = CrossConditionEarlyLayerStep(primary_threshold=5)
        self.assertEqual(step.primary_threshold, 5)

    def test_run_returns_none_when_no_data(self):
        """Test that run returns None when no data found."""
        condition_results = {"cond1": {}}

        step = CrossConditionEarlyLayerStep()
        result = step.run(condition_results)

        self.assertIsNone(result)

    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_mean_comparison')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_significance_effect_sizes')
    def test_run_with_single_threshold_returns_data(self, *mocks):
        """Test that run with single threshold returns combined data."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([2])
            },
            "cond2": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([2])
            },
        }

        step = CrossConditionEarlyLayerStep()
        result = step.run(condition_results)

        self.assertIsNotNone(result)
        self.assertIn('data', result)
        self.assertIn('statistics', result)

    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_mean_comparison')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_significance_effect_sizes')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_threshold_comparison')
    def test_run_with_multiple_thresholds_calls_threshold_comparison(self, mock_threshold, *other_mocks):
        """Test that multiple thresholds trigger threshold comparison plot."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([1, 2, 3])
            },
            "cond2": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([1, 2, 3])
            },
        }

        step = CrossConditionEarlyLayerStep()
        step.run(condition_results)

        mock_threshold.assert_called_once()

    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_boxplot')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_mean_comparison')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_significance_effect_sizes')
    @patch('src.analysis.cross_condition_analysis.cross_condition_early_layer_step.plot_early_layer_threshold_comparison')
    def test_run_with_single_threshold_skips_threshold_comparison(self, mock_threshold, *other_mocks):
        """Test that single threshold skips threshold comparison plot."""
        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([2])
            },
            "cond2": {
                SupportedConfigAnalyzeStep.EARLY_LAYER_CONTRIBUTION:
                    TestCrossConditionAnalyzerFixtures.create_early_layer_df([2])
            },
        }

        step = CrossConditionEarlyLayerStep()
        step.run(condition_results)

        mock_threshold.assert_not_called()


class TestCrossConditionL0Step(unittest.TestCase):
    """Tests for CrossConditionL0Step."""

    @staticmethod
    def create_l0_df() -> pd.DataFrame:
        """Create mock L0 DataFrame."""
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import (
            L0_VALUE_COL, L0_NORMALIZED_COL, LAYER_COL, PROMPT_ID_COL, D_TRANSCODER_COL
        )
        return pd.DataFrame({
            CONFIG_NAME_COL: ["config1", "config1", "config2", "config2"],
            PROMPT_ID_COL: ["p1", "p1", "p2", "p2"],
            LAYER_COL: [0, 1, 0, 1],
            L0_VALUE_COL: [100.0, 200.0, 150.0, 250.0],
            L0_NORMALIZED_COL: [0.1, 0.2, 0.15, 0.25],
            D_TRANSCODER_COL: [1000, 1000, 1000, 1000]
        })

    def test_config_results_key_is_l0_replacement_model(self):
        """Test that step uses L0_REPLACEMENT_MODEL key."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step

        step = CrossConditionL0Step()
        self.assertEqual(step.CONFIG_RESULTS_KEY, SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL)

    def test_results_sub_key_is_df(self):
        """Test that step uses 'df' sub-key."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step

        step = CrossConditionL0Step()
        self.assertEqual(step.RESULTS_SUB_KEY, "df")

    def test_run_returns_none_when_no_data(self):
        """Test that run returns None when no L0 data found."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step

        condition_results = {"cond1": {}}
        step = CrossConditionL0Step()
        result = step.run(condition_results)

        self.assertIsNone(result)

    @patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_by_condition')
    @patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_line')
    def test_run_returns_combined_dataframe(self, mock_line, mock_bar):
        """Test that run returns combined DataFrame."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step

        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: {
                    "df": self.create_l0_df()
                }
            },
            "cond2": {
                SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: {
                    "df": self.create_l0_df()
                }
            }
        }

        step = CrossConditionL0Step()
        result = step.run(condition_results)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn(step.CONDITION_COL, result.columns)

    @patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_by_condition')
    @patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_line')
    def test_run_generates_plots_for_raw_and_normalized(self, mock_line, mock_bar):
        """Test that run generates plots for both raw and normalized L0."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import CrossConditionL0Step

        condition_results = {
            "cond1": {
                SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL: {
                    "df": self.create_l0_df()
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CrossConditionL0Step(save_path=Path(tmpdir))
            step.run(condition_results)

            # Should call each plot function twice (raw and normalized)
            self.assertEqual(mock_bar.call_count, 2)
            self.assertEqual(mock_line.call_count, 2)

    def test_generate_l0_plots_helper(self):
        """Test _generate_l0_plots helper method."""
        from src.analysis.cross_condition_analysis.cross_condition_l0_step import (
            CrossConditionL0Step, L0_COMPARISON_FILENAME
        )
        from src.analysis.cross_config_analysis.cross_config_l0_replacement_model_step import L0_VALUE_COL

        with tempfile.TemporaryDirectory() as tmpdir:
            step = CrossConditionL0Step(save_path=Path(tmpdir))

            df = self.create_l0_df()
            df[step.CONDITION_COL] = "cond1"

            with patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_by_condition') as mock_bar, \
                 patch('src.analysis.cross_condition_analysis.cross_condition_l0_step.plot_l0_per_layer_line') as mock_line:

                step._generate_l0_plots(df, ["cond1"], L0_VALUE_COL, "test_prefix")

                mock_bar.assert_called_once()
                mock_line.assert_called_once()

                # Check filenames include prefix
                bar_call_kwargs = mock_bar.call_args[1]
                self.assertIn("test_prefix", str(bar_call_kwargs['save_path']))


if __name__ == '__main__':
    unittest.main()
