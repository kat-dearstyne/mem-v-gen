import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import torch

from src.analysis.config_analysis.config_analyzer import ConfigAnalyzer, STEP2CLASS
from src.analysis.config_analysis.config_error_ranking_step import (
    ConfigErrorRankingStep, MannWhitneyResult, PermutationTestResult
)
from src.analysis.config_analysis.config_subgraph_filter_step import ConfigSubgraphFilterStep
from src.analysis.config_analysis.config_feature_overlap_step import ConfigFeatureOverlapStep
from src.analysis.config_analysis.config_token_subgraph_step import ConfigTokenSubgraphStep
from src.analysis.config_analysis.config_replacement_model_accuracy_step import ConfigReplacementModelAccuracyStep
from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.graph_manager import GraphManager, Feature
from src.metrics import (
    ErrorRankingMetrics, ComparisonMetrics, FeatureSharingMetrics, ReplacementAccuracyMetrics
)


class TestConfigAnalyzerFixtures:
    """Shared test fixtures for ConfigAnalyzer tests."""

    @staticmethod
    def create_graph_metadata(prompt_id: str, prompt: str, target_token: str = "hello"):
        """Create graph metadata for testing."""
        return {
            "metadata": {
                "slug": f"test-{prompt_id}",
                "scan": "gemma-2-2b",
                "prompt": prompt,
                "info": {"url": f"https://example.com/graph/{prompt_id}"}
            },
            "nodes": [
                {
                    "node_id": "0_1000_1",
                    "feature": 1000,
                    "layer": "0",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "1_2000_1",
                    "feature": 2000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": f"logit_5000_1",
                    "feature": 5000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.95,
                    "is_target_logit": True,
                    "clerp": f'"{target_token}"'
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "1_2000_1", "weight": 3.0},
                {"source": "1_2000_1", "target": "logit_5000_1", "weight": 8.0}
            ]
        }

    @staticmethod
    def create_mock_graph_analyzer():
        """Create a mock GraphAnalyzer for testing."""
        mock_analyzer = MagicMock()

        # Create mock graphs
        graph1 = GraphManager(TestConfigAnalyzerFixtures.create_graph_metadata("main", "main prompt"))
        graph2 = GraphManager(TestConfigAnalyzerFixtures.create_graph_metadata("other", "other prompt"))

        mock_analyzer.graphs = {"main": graph1, "other": graph2}
        mock_analyzer.dfs = {
            "main": graph1.create_node_df(exclude_embeddings=True, exclude_errors=True,
                                          exclude_logits=True, drop_duplicates=True),
            "other": graph2.create_node_df(exclude_embeddings=True, exclude_errors=True,
                                           exclude_logits=True, drop_duplicates=True)
        }
        mock_analyzer.prompts = {"main": "main prompt", "other": "other prompt"}

        # Mock get_graph_and_df
        def mock_get_graph_and_df(prompt_id):
            return mock_analyzer.graphs[prompt_id], mock_analyzer.dfs[prompt_id]

        mock_analyzer.get_graph_and_df = mock_get_graph_and_df

        # Mock neuronpedia_manager
        mock_analyzer.neuronpedia_manager = MagicMock()
        mock_analyzer.neuronpedia_manager.filter_features_for_subgraph.side_effect = lambda df, *args, **kwargs: df
        mock_analyzer.neuronpedia_manager.create_subgraph_from_selected_features.return_value = None

        return mock_analyzer


class TestConfigAnalyzerInit(unittest.TestCase):
    """Tests for ConfigAnalyzer initialization."""

    @patch('src.analysis.config_analysis.config_analyzer.GraphAnalyzer')
    def test_init_creates_graph_analyzer(self, mock_graph_analyzer_cls):
        """Test that init creates a GraphAnalyzer."""
        mock_npm = MagicMock()
        prompts = {"main": "test prompt"}

        analyzer = ConfigAnalyzer(neuronpedia_manager=mock_npm, prompts=prompts)

        mock_graph_analyzer_cls.assert_called_once_with(prompts=prompts, neuronpedia_manager=mock_npm)


class TestConfigAnalyzerRun(unittest.TestCase):
    """Tests for ConfigAnalyzer.run method."""

    def setUp(self):
        self.mock_npm = MagicMock()
        self.prompts = {"main": "test prompt", "other": "other prompt"}

    @patch('src.analysis.config_analysis.config_analyzer.GraphAnalyzer')
    def test_run_single_step(self, mock_graph_analyzer_cls):
        """Test running a single analysis step."""
        mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()
        mock_graph_analyzer_cls.return_value = mock_graph_analyzer

        # Mock the step class
        mock_step_cls = MagicMock()
        mock_step_cls.__name__ = "MockStep"
        mock_step_instance = MagicMock()
        mock_step_instance.run.return_value = {"result": "test"}
        mock_step_cls.return_value = mock_step_instance

        with patch.dict(STEP2CLASS, {SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: mock_step_cls}):
            analyzer = ConfigAnalyzer(neuronpedia_manager=self.mock_npm, prompts=self.prompts)
            results = analyzer.run(SupportedConfigAnalyzeStep.SUBGRAPH_FILTER, main_prompt_id="main")

            self.assertIn(SupportedConfigAnalyzeStep.SUBGRAPH_FILTER, results)
            self.assertEqual(results[SupportedConfigAnalyzeStep.SUBGRAPH_FILTER], {"result": "test"})

    @patch('src.analysis.config_analysis.config_analyzer.GraphAnalyzer')
    def test_run_multiple_steps(self, mock_graph_analyzer_cls):
        """Test running multiple analysis steps."""
        mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()
        mock_graph_analyzer_cls.return_value = mock_graph_analyzer

        # Mock step classes with __name__ attribute
        mock_step1_cls = MagicMock()
        mock_step1_cls.__name__ = "MockStep1"
        mock_step1_instance = MagicMock()
        mock_step1_instance.run.return_value = {"step1": "result1"}
        mock_step1_cls.return_value = mock_step1_instance

        mock_step2_cls = MagicMock()
        mock_step2_cls.__name__ = "MockStep2"
        mock_step2_instance = MagicMock()
        mock_step2_instance.run.return_value = {"step2": "result2"}
        mock_step2_cls.return_value = mock_step2_instance

        with patch.dict(STEP2CLASS, {
            SupportedConfigAnalyzeStep.SUBGRAPH_FILTER: mock_step1_cls,
            SupportedConfigAnalyzeStep.ERROR_RANKING: mock_step2_cls
        }):
            analyzer = ConfigAnalyzer(neuronpedia_manager=self.mock_npm, prompts=self.prompts)
            results = analyzer.run(
                [SupportedConfigAnalyzeStep.SUBGRAPH_FILTER, SupportedConfigAnalyzeStep.ERROR_RANKING],
                main_prompt_id="main"
            )

            self.assertEqual(len(results), 2)
            self.assertIn(SupportedConfigAnalyzeStep.SUBGRAPH_FILTER, results)
            self.assertIn(SupportedConfigAnalyzeStep.ERROR_RANKING, results)

    @patch('src.analysis.config_analysis.config_analyzer.GraphAnalyzer')
    def test_run_raises_for_unknown_step(self, mock_graph_analyzer_cls):
        """Test that run raises for unknown step type."""
        mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()
        mock_graph_analyzer_cls.return_value = mock_graph_analyzer

        # Create a fake step type not in STEP2CLASS
        analyzer = ConfigAnalyzer(neuronpedia_manager=self.mock_npm, prompts=self.prompts)

        # FEATURE_PRESENCE is in the enum but not mapped in STEP2CLASS
        with self.assertRaises(AssertionError):
            analyzer.run(SupportedConfigAnalyzeStep.FEATURE_PRESENCE)


class TestConfigErrorRankingStep(unittest.TestCase):
    """Tests for ConfigErrorRankingStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_is_error_node_true(self):
        """Test is_error_node returns True for error nodes."""
        error_node = {"feature_type": "mlp reconstruction error"}
        self.assertTrue(ConfigErrorRankingStep.is_error_node(error_node))

    def test_is_error_node_false(self):
        """Test is_error_node returns False for non-error nodes."""
        regular_node = {"feature_type": "cross layer transcoder"}
        self.assertFalse(ConfigErrorRankingStep.is_error_node(regular_node))

    def test_get_relevance_list(self):
        """Test get_relevance_list creates binary relevance list."""
        nodes = [
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "mlp reconstruction error"},
        ]

        relevance = ConfigErrorRankingStep.get_relevance_list(nodes)

        self.assertEqual(relevance, [0, 1, 0, 1])

    def test_get_error_percentile_ranks(self):
        """Test get_error_percentile_ranks computes correct percentiles."""
        nodes = [
            {"feature_type": "mlp reconstruction error"},  # rank 1/4 = 0.25
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "mlp reconstruction error"},  # rank 3/4 = 0.75
            {"feature_type": "cross layer transcoder"},
        ]

        percentiles = ConfigErrorRankingStep.get_error_percentile_ranks(nodes)

        self.assertEqual(len(percentiles), 2)
        self.assertAlmostEqual(percentiles[0], 0.25, places=2)
        self.assertAlmostEqual(percentiles[1], 0.75, places=2)

    def test_top_k_error_proportion(self):
        """Test top_k_error_proportion calculates correct proportion."""
        nodes = [
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "cross layer transcoder"},
        ]

        # Top 2 has 2 errors = 100%
        prop_k2 = ConfigErrorRankingStep.top_k_error_proportion(nodes, k=2)
        self.assertEqual(prop_k2, 1.0)

        # Top 4 has 2 errors = 50%
        prop_k4 = ConfigErrorRankingStep.top_k_error_proportion(nodes, k=4)
        self.assertEqual(prop_k4, 0.5)

    def test_ndcg_at_k_perfect_ranking(self):
        """Test NDCG is 1.0 for perfect ranking (all errors at top)."""
        nodes = [
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "cross layer transcoder"},
        ]

        ndcg = ConfigErrorRankingStep.ndcg_at_k(nodes, k=4)

        self.assertEqual(ndcg, 1.0)

    def test_ndcg_at_k_worst_ranking(self):
        """Test NDCG is less than 1 for non-ideal ranking."""
        nodes = [
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "mlp reconstruction error"},
        ]

        ndcg = ConfigErrorRankingStep.ndcg_at_k(nodes, k=4)

        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)

    def test_average_precision_perfect(self):
        """Test average precision is 1.0 for perfect ranking."""
        nodes = [
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "mlp reconstruction error"},
            {"feature_type": "cross layer transcoder"},
        ]

        ap = ConfigErrorRankingStep.average_precision(nodes)

        self.assertEqual(ap, 1.0)

    def test_average_precision_mixed(self):
        """Test average precision for mixed ranking."""
        nodes = [
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "mlp reconstruction error"},  # P@2 = 1/2
            {"feature_type": "mlp reconstruction error"},  # P@3 = 2/3
            {"feature_type": "cross layer transcoder"},
        ]

        # AP = (1/2 + 2/3) / 2 = 7/12 ≈ 0.583
        ap = ConfigErrorRankingStep.average_precision(nodes)

        self.assertAlmostEqual(ap, 7/12, places=2)

    def test_average_precision_no_errors(self):
        """Test average precision is 0 when no errors."""
        nodes = [
            {"feature_type": "cross layer transcoder"},
            {"feature_type": "cross layer transcoder"},
        ]

        ap = ConfigErrorRankingStep.average_precision(nodes)

        self.assertEqual(ap, 0.0)


class TestConfigSubgraphFilterStep(unittest.TestCase):
    """Tests for ConfigSubgraphFilterStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_init_stores_parameters(self):
        """Test that init stores all parameters correctly."""
        step = ConfigSubgraphFilterStep(
            graph_analyzer=self.mock_graph_analyzer,
            main_prompt_id="main",
            prompts_with_shared_features=["shared1"],
            prompts_with_unique_features=["unique1"],
            metrics2run={ComparisonMetrics.JACCARD_INDEX},
            create_subgraph=False
        )

        self.assertEqual(step.main_prompt_id, "main")
        self.assertEqual(step.prompts_with_shared_features, ["shared1"])
        self.assertEqual(step.prompts_with_unique_features, ["unique1"])
        self.assertEqual(step.metrics2run, {ComparisonMetrics.JACCARD_INDEX})
        self.assertFalse(step.create_subgraph)

    def test_run_with_unique_features_only(self):
        """Test run with only prompts_with_unique_features."""
        # Setup mock return values
        unique_df = pd.DataFrame({"layer": ["0"], "feature": ["1000"]})
        self.mock_graph_analyzer.nodes_not_in.return_value = (unique_df, {"diff_metrics": "value"})

        step = ConfigSubgraphFilterStep(
            graph_analyzer=self.mock_graph_analyzer,
            main_prompt_id="main",
            prompts_with_unique_features=["other"],
            create_subgraph=False
        )

        results = step.run()

        self.assertIn("diff", results)
        self.mock_graph_analyzer.nodes_not_in.assert_called_once()

    def test_run_with_shared_features_only(self):
        """Test run with only prompts_with_shared_features."""
        shared_df = pd.DataFrame({"layer": ["0"], "feature": ["1000"]})
        self.mock_graph_analyzer.nodes_in.return_value = (shared_df, {"sim_metrics": "value"})

        step = ConfigSubgraphFilterStep(
            graph_analyzer=self.mock_graph_analyzer,
            main_prompt_id="main",
            prompts_with_shared_features=["other"],
            create_subgraph=False
        )

        results = step.run()

        self.assertIn("sim", results)
        self.mock_graph_analyzer.nodes_in.assert_called_once()


class TestConfigFeatureOverlapStep(unittest.TestCase):
    """Tests for ConfigFeatureOverlapStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_init_stores_parameters(self):
        """Test that init stores all parameters correctly."""
        step = ConfigFeatureOverlapStep(
            graph_analyzer=self.mock_graph_analyzer,
            main_prompt_id="main",
            comparison_prompt_ids=["other"],
            debug=True,
            create_subgraphs=False,
            filter_by_act_density=50
        )

        self.assertEqual(step.main_prompt_id, "main")
        self.assertEqual(step.comparison_prompt_ids, ["other"])
        self.assertTrue(step.debug)
        self.assertFalse(step.create_subgraphs)
        self.assertEqual(step.filter_by_act_density, 50)

    def test_run_returns_combined_metrics(self):
        """Test that run returns combined unique, shared, and edge metrics."""
        # Setup mock return values
        unique_df = pd.DataFrame({"layer": ["2"], "feature": ["3000"]})
        shared_df = pd.DataFrame({"layer": ["0", "1"], "feature": ["1000", "2000"]})

        unique_metrics = {
            FeatureSharingMetrics.UNIQUE_FRAC: 0.33,
            FeatureSharingMetrics.NUM_UNIQUE: 1,
            FeatureSharingMetrics.NUM_MAIN: 3
        }
        shared_metrics = {
            FeatureSharingMetrics.SHARED_FRAC: 0.67,
            FeatureSharingMetrics.NUM_SHARED: 2
        }
        edge_metrics = {
            FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC: 0.4,
            FeatureSharingMetrics.SHARED_WEIGHTED_FRAC: 0.6
        }

        self.mock_graph_analyzer.nodes_not_in.return_value = (unique_df, unique_metrics)
        self.mock_graph_analyzer.nodes_in.return_value = (shared_df, shared_metrics)
        self.mock_graph_analyzer.calculate_edge_sharing_metrics.return_value = edge_metrics
        self.mock_graph_analyzer.get_most_frequent_features.return_value = pd.DataFrame()

        step = ConfigFeatureOverlapStep(
            graph_analyzer=self.mock_graph_analyzer,
            main_prompt_id="main",
            comparison_prompt_ids=["other"],
            create_subgraphs=False
        )

        results = step.run()

        # Check all metrics are present
        self.assertIn(FeatureSharingMetrics.UNIQUE_FRAC, results)
        self.assertIn(FeatureSharingMetrics.SHARED_FRAC, results)
        self.assertIn(FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC, results)


class TestConfigTokenSubgraphStep(unittest.TestCase):
    """Tests for ConfigTokenSubgraphStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_init_stores_parameters(self):
        """Test that init stores all parameters correctly."""
        step = ConfigTokenSubgraphStep(
            graph_analyzer=self.mock_graph_analyzer,
            prompt_id="main",
            token_of_interest="hello",
            create_subgraph=False
        )

        self.assertEqual(step.prompt_id, "main")
        self.assertEqual(step.token_of_interest, "hello")
        self.assertFalse(step.create_subgraph)

    def test_run_returns_dataframe(self):
        """Test that run returns a DataFrame of unique features."""
        # Mock compare_token_subgraphs to return a list of Features
        unique_features = [Feature("2", "3000"), Feature("3", "4000")]
        self.mock_graph_analyzer.compare_token_subgraphs.return_value = unique_features

        step = ConfigTokenSubgraphStep(
            graph_analyzer=self.mock_graph_analyzer,
            prompt_id="main",
            token_of_interest="hello",
            create_subgraph=False
        )

        results = step.run()

        self.assertIsInstance(results, pd.DataFrame)
        # Should only include features at layers > 1
        self.assertEqual(len(results), 2)

    def test_run_filters_low_layers(self):
        """Test that run filters out features at layer <= 1."""
        unique_features = [
            Feature("0", "1000"),  # Should be filtered
            Feature("1", "2000"),  # Should be filtered
            Feature("2", "3000"),  # Should be included
        ]
        self.mock_graph_analyzer.compare_token_subgraphs.return_value = unique_features

        step = ConfigTokenSubgraphStep(
            graph_analyzer=self.mock_graph_analyzer,
            prompt_id="main",
            token_of_interest="hello",
            create_subgraph=False
        )

        results = step.run()

        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]["layer"], "2")


class TestConfigReplacementModelAccuracyStep(unittest.TestCase):
    """Tests for ConfigReplacementModelAccuracyStep static methods."""

    def test_get_per_position_cosine(self):
        """Test per-position cosine similarity calculation."""
        # Create test tensors
        base_logits = torch.randn(1, 3, 100)
        replacement_logits = base_logits.clone()  # Same logits should give cosine ~1

        cosines = ConfigReplacementModelAccuracyStep.get_per_position_cosine(
            base_logits, replacement_logits
        )

        self.assertEqual(len(cosines), 3)
        # Same tensors should have cosine similarity close to 1
        for cosine in cosines:
            self.assertAlmostEqual(cosine, 1.0, places=4)

    def test_get_per_position_cosine_orthogonal(self):
        """Test cosine similarity for different tensors."""
        base_logits = torch.randn(1, 3, 100)
        replacement_logits = torch.randn(1, 3, 100)  # Different random tensors

        cosines = ConfigReplacementModelAccuracyStep.get_per_position_cosine(
            base_logits, replacement_logits
        )

        self.assertEqual(len(cosines), 3)
        # Random tensors should have lower cosine similarity
        for cosine in cosines:
            self.assertLessEqual(cosine, 1.0)
            self.assertGreaterEqual(cosine, 0.0)

    def test_get_last_token_accuracy(self):
        """Test last token accuracy (cosine similarity)."""
        base_logits = torch.randn(1, 5, 100)
        replacement_logits = base_logits.clone()

        accuracy = ConfigReplacementModelAccuracyStep.get_last_token_accuracy(
            base_logits, replacement_logits
        )

        self.assertAlmostEqual(accuracy, 1.0, places=4)

    def test_get_per_position_argmax_match(self):
        """Test per-position argmax matching."""
        base_logits = torch.zeros(1, 3, 10)
        base_logits[0, 0, 5] = 1.0  # argmax = 5
        base_logits[0, 1, 3] = 1.0  # argmax = 3
        base_logits[0, 2, 7] = 1.0  # argmax = 7

        replacement_logits = torch.zeros(1, 3, 10)
        replacement_logits[0, 0, 5] = 1.0  # match
        replacement_logits[0, 1, 8] = 1.0  # no match
        replacement_logits[0, 2, 7] = 1.0  # match

        matches = ConfigReplacementModelAccuracyStep.get_per_position_argmax_match(
            base_logits, replacement_logits
        )

        self.assertEqual(matches, [1, 0, 1])

    def test_get_original_accuracy_metric(self):
        """Test original accuracy metric calculation."""
        base_logits = torch.zeros(1, 4, 10)
        base_logits[0, 0, 5] = 1.0
        base_logits[0, 1, 3] = 1.0
        base_logits[0, 2, 7] = 1.0
        base_logits[0, 3, 2] = 1.0

        replacement_logits = torch.zeros(1, 4, 10)
        replacement_logits[0, 0, 5] = 1.0  # match
        replacement_logits[0, 1, 8] = 1.0  # no match
        replacement_logits[0, 2, 7] = 1.0  # match
        replacement_logits[0, 3, 2] = 1.0  # match

        prompt_tokens = torch.zeros(1, 4)

        accuracy = ConfigReplacementModelAccuracyStep.get_original_accuracy_metric(
            base_logits, replacement_logits, prompt_tokens
        )

        # 3 out of 4 positions match
        self.assertAlmostEqual(accuracy, 0.75, places=2)

    def test_get_kl_divergence_same_distribution(self):
        """Test KL divergence is 0 for identical distributions."""
        logits = torch.randn(1, 3, 100)

        kl_div = ConfigReplacementModelAccuracyStep.get_kl_divergence(logits, logits)

        self.assertAlmostEqual(kl_div, 0.0, places=4)

    def test_get_kl_divergence_different_distributions(self):
        """Test KL divergence is positive for different distributions."""
        base_logits = torch.randn(1, 3, 100)
        replacement_logits = torch.randn(1, 3, 100)

        kl_div = ConfigReplacementModelAccuracyStep.get_kl_divergence(
            base_logits, replacement_logits
        )

        self.assertGreater(kl_div, 0.0)

    def test_get_top_k_agreement_same(self):
        """Test top-k agreement is 1.0 for identical distributions."""
        logits = torch.randn(1, 3, 100)

        agreement = ConfigReplacementModelAccuracyStep.get_top_k_agreement(logits, logits, k=5)

        self.assertEqual(agreement, 1.0)

    def test_get_top_k_agreement_different(self):
        """Test top-k agreement for different distributions."""
        base_logits = torch.randn(1, 3, 1000)
        replacement_logits = torch.randn(1, 3, 1000)

        agreement = ConfigReplacementModelAccuracyStep.get_top_k_agreement(
            base_logits, replacement_logits, k=5
        )

        # Should be between 0 and 1
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)

    def test_get_replacement_prob_of_original_top(self):
        """Test getting replacement probability of original top token."""
        base_logits = torch.zeros(1, 3, 10)
        base_logits[0, -1, 5] = 10.0  # Strong preference for token 5

        replacement_logits = torch.zeros(1, 3, 10)
        replacement_logits[0, -1, 5] = 10.0  # Also strong preference for token 5

        prob = ConfigReplacementModelAccuracyStep.get_replacement_prob_of_original_top(
            base_logits, replacement_logits
        )

        # Should be close to 1 since both strongly prefer token 5
        self.assertGreater(prob, 0.9)

    def test_get_per_position_kl_divergence(self):
        """Test per-position KL divergence calculation."""
        base_logits = torch.randn(1, 3, 100)
        replacement_logits = torch.randn(1, 3, 100)

        kl_divs = ConfigReplacementModelAccuracyStep.get_per_position_kl_divergence(
            base_logits, replacement_logits
        )

        self.assertEqual(len(kl_divs), 3)
        # KL divergence should be non-negative
        for kl in kl_divs:
            self.assertGreaterEqual(kl, 0.0)


class TestSupportedConfigAnalyzeStepMapping(unittest.TestCase):
    """Tests for STEP2CLASS mapping."""

    def test_all_mapped_steps_have_classes(self):
        """Test that all mapped steps have valid class entries."""
        for step_type, step_cls in STEP2CLASS.items():
            self.assertIsNotNone(step_cls)
            self.assertTrue(hasattr(step_cls, 'run'))

    def test_expected_steps_are_mapped(self):
        """Test that expected steps are in the mapping."""
        expected_steps = [
            SupportedConfigAnalyzeStep.ERROR_RANKING,
            SupportedConfigAnalyzeStep.FEATURE_OVERLAP,
            SupportedConfigAnalyzeStep.REPLACEMENT_MODEL,
            SupportedConfigAnalyzeStep.SUBGRAPH_FILTER,
            SupportedConfigAnalyzeStep.TOKEN_SUBGRAPH
        ]

        for step in expected_steps:
            self.assertIn(step, STEP2CLASS)


class TestConfigL0ReplacementModelStep(unittest.TestCase):
    """Tests for ConfigL0ReplacementModelStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_get_test_output_returns_tensor(self):
        """Test that _get_test_output returns a tensor of zeros."""
        from src.analysis.config_analysis.config_l0_replacement_model_step import ConfigL0ReplacementModelStep

        result = ConfigL0ReplacementModelStep._get_test_output()

        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.sum().item(), 0.0)

    @patch('src.analysis.config_analysis.config_l0_replacement_model_step.IS_TEST', True)
    def test_compute_l0_for_prompt_in_test_mode(self):
        """Test compute_l0_for_prompt returns dummy output in test mode."""
        from src.analysis.config_analysis.config_l0_replacement_model_step import ConfigL0ReplacementModelStep

        step = ConfigL0ReplacementModelStep(graph_analyzer=self.mock_graph_analyzer)
        result = step.compute_l0_for_prompt("test prompt")

        self.assertIsInstance(result, torch.Tensor)

    @patch('src.analysis.config_analysis.config_l0_replacement_model_step.IS_TEST', True)
    def test_run_returns_results_dict(self):
        """Test that run returns dict with results and d_transcoder keys."""
        from src.analysis.config_analysis.config_l0_replacement_model_step import ConfigL0ReplacementModelStep

        step = ConfigL0ReplacementModelStep(graph_analyzer=self.mock_graph_analyzer)
        result = step.run()

        self.assertIn("results", result)
        self.assertIn("d_transcoder", result)
        self.assertIsInstance(result["results"], dict)


class TestConfigEarlyLayerContributionStep(unittest.TestCase):
    """Tests for ConfigEarlyLayerContributionStep."""

    def setUp(self):
        self.mock_graph_analyzer = TestConfigAnalyzerFixtures.create_mock_graph_analyzer()

    def test_init_stores_max_layer(self):
        """Test that init stores max_layer parameter."""
        from src.analysis.config_analysis.config_early_layer_contribution_step import ConfigEarlyLayerContributionStep

        step = ConfigEarlyLayerContributionStep(
            graph_analyzer=self.mock_graph_analyzer,
            max_layer=5
        )

        self.assertEqual(step.max_layer, 5)

    def test_init_max_layer_defaults_to_none(self):
        """Test that max_layer defaults to None."""
        from src.analysis.config_analysis.config_early_layer_contribution_step import ConfigEarlyLayerContributionStep

        step = ConfigEarlyLayerContributionStep(graph_analyzer=self.mock_graph_analyzer)

        self.assertIsNone(step.max_layer)

    def test_run_with_specific_max_layer(self):
        """Test run with specific max_layer."""
        from src.analysis.config_analysis.config_early_layer_contribution_step import ConfigEarlyLayerContributionStep

        # Mock the early layer contribution method
        self.mock_graph_analyzer.get_early_layer_contribution_fraction = MagicMock(return_value=0.5)

        step = ConfigEarlyLayerContributionStep(
            graph_analyzer=self.mock_graph_analyzer,
            max_layer=2
        )

        results = step.run()

        # Should have results for each prompt
        self.assertIn("main", results)
        self.assertIn("other", results)

        # Each should have only the specified max_layer
        self.assertIn(2, results["main"])
        self.assertEqual(results["main"][2], 0.5)


if __name__ == "__main__":
    unittest.main()
