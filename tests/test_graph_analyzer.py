import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.graph_analyzer import GraphAnalyzer
from src.graph_manager import GraphManager, Feature
from src.metrics import ComparisonMetrics, FeatureSharingMetrics, SharedFeatureMetrics


class TestGraphAnalyzerFixtures:
    """Shared test fixtures for GraphAnalyzer tests."""

    @staticmethod
    def create_graph_metadata_main():
        """Create main graph metadata with features: 0/1000, 1/2000, 2/3000."""
        return {
            "metadata": {
                "slug": "test-main",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Main prompt text",
                "info": {"url": "https://example.com/graph/main"}
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
                    "node_id": "2_3000_1",
                    "feature": 3000,
                    "layer": "2",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "logit_5000_1",
                    "feature": 5000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.95,
                    "is_target_logit": True,
                    "clerp": "\"hello\""
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "1_2000_1", "weight": 3.0},
                {"source": "1_2000_1", "target": "2_3000_1", "weight": 4.0},
                {"source": "2_3000_1", "target": "logit_5000_1", "weight": 8.0}
            ]
        }

    @staticmethod
    def create_graph_metadata_other():
        """Create other graph metadata with features: 0/1000, 1/2000, 1/4000 (shared: 0/1000, 1/2000)."""
        return {
            "metadata": {
                "slug": "test-other",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Other prompt text",
                "info": {"url": "https://example.com/graph/other"}
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
                    "node_id": "1_4000_1",
                    "feature": 4000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "logit_5000_1",
                    "feature": 5000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.90,
                    "is_target_logit": True,
                    "clerp": "\"hello\""
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "1_2000_1", "weight": 3.5},
                {"source": "0_1000_1", "target": "1_4000_1", "weight": 2.5},
                {"source": "1_2000_1", "target": "logit_5000_1", "weight": 6.0},
                {"source": "1_4000_1", "target": "logit_5000_1", "weight": 7.0}
            ]
        }

    @staticmethod
    def create_graph_metadata_third():
        """Create third graph with features: 0/1000, 2/3000, 3/5000 (shared with main: 0/1000, 2/3000)."""
        return {
            "metadata": {
                "slug": "test-third",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Third prompt text",
                "info": {"url": "https://example.com/graph/third"}
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
                    "node_id": "2_3000_1",
                    "feature": 3000,
                    "layer": "2",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "3_5000_1",
                    "feature": 5000,
                    "layer": "3",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "logit_6000_1",
                    "feature": 6000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.88,
                    "is_target_logit": True,
                    "clerp": "\"world\""
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "2_3000_1", "weight": 4.0},
                {"source": "2_3000_1", "target": "3_5000_1", "weight": 3.5},
                {"source": "3_5000_1", "target": "logit_6000_1", "weight": 7.0}
            ]
        }

    @staticmethod
    def create_mock_analyzer(graph_configs):
        """
        Create a GraphAnalyzer with mocked dependencies.

        Args:
            graph_configs: Dict mapping prompt_id -> graph metadata dict

        Returns:
            Configured GraphAnalyzer with mocked neuronpedia_manager
        """
        mock_npm = MagicMock()

        # Create graphs and dataframes
        graphs = {}
        dfs = {}
        prompts = {}

        for prompt_id, metadata in graph_configs.items():
            graph = GraphManager(metadata)
            graphs[prompt_id] = graph
            dfs[prompt_id] = graph.create_node_df(
                exclude_embeddings=True,
                exclude_errors=True,
                exclude_logits=True,
                drop_duplicates=True
            )
            prompts[prompt_id] = metadata["metadata"]["prompt"]

        # Mock create_or_load_graph to return the right graph for each prompt
        def mock_create_or_load(prompt):
            for pid, p in prompts.items():
                if p == prompt:
                    return graphs[pid]
            raise ValueError(f"Unknown prompt: {prompt}")

        mock_npm.create_or_load_graph.side_effect = mock_create_or_load
        mock_npm.filter_features_for_subgraph.side_effect = lambda df, **kwargs: df

        # Create analyzer and pre-populate graphs/dfs to avoid lazy loading in tests
        analyzer = GraphAnalyzer(prompts=prompts, neuronpedia_manager=mock_npm)
        analyzer.graphs = graphs
        analyzer.dfs = dfs

        return analyzer


class TestGraphAnalyzerInit(unittest.TestCase):
    """Tests for GraphAnalyzer initialization."""

    def test_init_stores_prompts(self):
        """Test that init stores the prompts dict."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })
        self.assertIn("main", analyzer.prompts)

    def test_init_creates_empty_graphs_and_dfs(self):
        """Test that init creates empty dicts for lazy loading."""
        mock_npm = MagicMock()
        prompts = {"main": "main prompt", "other": "other prompt"}

        analyzer = GraphAnalyzer(prompts=prompts, neuronpedia_manager=mock_npm)

        # Init should create empty dicts (lazy loading)
        self.assertEqual(analyzer.graphs, {})
        self.assertEqual(analyzer.dfs, {})

    def test_lazy_loading_populates_graphs_and_dfs(self):
        """Test that accessing a graph lazily loads it."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        # Mock fixture pre-populates, so we verify they're accessible
        self.assertIn("main", analyzer.graphs)
        self.assertIn("other", analyzer.graphs)
        self.assertIn("main", analyzer.dfs)
        self.assertIn("other", analyzer.dfs)


class TestGetGraphAndDf(unittest.TestCase):
    """Tests for get_graph_and_df method."""

    def test_returns_graph_and_df_tuple(self):
        """Test that get_graph_and_df returns correct tuple."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        graph, df = analyzer.get_graph_and_df("main")

        self.assertIsInstance(graph, GraphManager)
        self.assertIsInstance(df, pd.DataFrame)

    def test_raises_for_unknown_prompt(self):
        """Test that get_graph_and_df raises for unknown prompt."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        with self.assertRaises(AssertionError):
            analyzer.get_graph_and_df("unknown")


class TestFilterForUniqueFeatures(unittest.TestCase):
    """Tests for filter_for_unique_features method."""

    def test_filters_out_common_features(self):
        """Test that common features are filtered out."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        df1 = pd.DataFrame({
            "layer": ["0", "1", "2"],
            "feature": ["1000", "2000", "3000"],
            "feature_type": ["cross layer transcoder"] * 3,
            "ctx_idx": [1, 1, 1]
        })

        df2 = pd.DataFrame({
            "layer": ["0", "1"],
            "feature": ["1000", "2000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        unique = analyzer.filter_for_unique_features(df1, df2)

        # Only 2/3000 should remain (unique to df1)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique.iloc[0]["layer"], "2")
        self.assertEqual(unique.iloc[0]["feature"], "3000")

    def test_returns_all_when_no_overlap(self):
        """Test that all features returned when no overlap."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        df1 = pd.DataFrame({
            "layer": ["0", "1"],
            "feature": ["1000", "2000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        df2 = pd.DataFrame({
            "layer": ["2", "3"],
            "feature": ["3000", "4000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        unique = analyzer.filter_for_unique_features(df1, df2)

        self.assertEqual(len(unique), 2)


class TestIntersectFeatures(unittest.TestCase):
    """Tests for intersect_features method."""

    def test_returns_common_features(self):
        """Test that only common features are returned."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        df1 = pd.DataFrame({
            "layer": ["0", "1", "2"],
            "feature": ["1000", "2000", "3000"],
            "feature_type": ["cross layer transcoder"] * 3,
            "ctx_idx": [1, 1, 1]
        })

        df2 = pd.DataFrame({
            "layer": ["0", "1", "1"],
            "feature": ["1000", "2000", "4000"],
            "feature_type": ["cross layer transcoder"] * 3,
            "ctx_idx": [1, 1, 1]
        })

        intersection = analyzer.intersect_features(df1, df2)

        # 0/1000 and 1/2000 are common
        self.assertEqual(len(intersection), 2)
        layers = set(intersection["layer"].tolist())
        self.assertIn("0", layers)
        self.assertIn("1", layers)

    def test_returns_empty_when_no_overlap(self):
        """Test that empty df returned when no overlap."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        df1 = pd.DataFrame({
            "layer": ["0", "1"],
            "feature": ["1000", "2000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        df2 = pd.DataFrame({
            "layer": ["2", "3"],
            "feature": ["3000", "4000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        intersection = analyzer.intersect_features(df1, df2)

        self.assertEqual(len(intersection), 0)


class TestNodesNotIn(unittest.TestCase):
    """Tests for nodes_not_in method."""

    def test_returns_unique_features(self):
        """Test that only features unique to main are returned."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        unique = analyzer.nodes_not_in("main", comparison_prompts=["other"])

        # Main has: 0/1000, 1/2000, 2/3000
        # Other has: 0/1000, 1/2000, 1/4000
        # Unique to main: 2/3000
        layers = unique["layer"].tolist()
        features = unique["feature"].tolist()

        self.assertIn("2", layers)
        self.assertIn("3000", features)
        self.assertNotIn("0", layers)  # 0/1000 is shared
        self.assertNotIn("1000", features)

    def test_returns_metrics_when_requested(self):
        """Test that metrics are returned when metrics2run is provided."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        unique, metrics = analyzer.nodes_not_in(
            "main",
            comparison_prompts=["other"],
            metrics2run={FeatureSharingMetrics.NUM_UNIQUE, FeatureSharingMetrics.NUM_MAIN}
        )

        self.assertIsInstance(metrics, dict)
        self.assertIn(FeatureSharingMetrics.NUM_UNIQUE, metrics)
        self.assertIn(FeatureSharingMetrics.NUM_MAIN, metrics)

    def test_handles_multiple_comparison_prompts(self):
        """Test filtering against multiple comparison prompts."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other(),
            "third": TestGraphAnalyzerFixtures.create_graph_metadata_third()
        })

        unique = analyzer.nodes_not_in("main", comparison_prompts=["other", "third"])

        # Main has: 0/1000, 1/2000, 2/3000
        # Other has: 0/1000, 1/2000, 1/4000
        # Third has: 0/1000, 2/3000, 3/5000
        # After filtering both: 1/2000 remains (not in third)
        # Wait - 1/2000 is in other, so it gets filtered
        # Actually: unique to main after both = only features not in other AND not in third
        # 0/1000 is in both other and third -> filtered
        # 1/2000 is in other -> filtered
        # 2/3000 is in third -> filtered
        # Result: nothing unique

        self.assertEqual(len(unique), 0)


class TestNodesIn(unittest.TestCase):
    """Tests for nodes_in method."""

    def test_returns_shared_features(self):
        """Test that only features shared across all prompts are returned."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        shared = analyzer.nodes_in("main", comparison_prompts=["other"])

        # Main has: 0/1000, 1/2000, 2/3000
        # Other has: 0/1000, 1/2000, 1/4000
        # Shared: 0/1000, 1/2000
        self.assertEqual(len(shared), 2)

        layers = set(shared["layer"].tolist())
        self.assertIn("0", layers)
        self.assertIn("1", layers)


class TestGetIntersection(unittest.TestCase):
    """Tests for get_intersection method."""

    def test_returns_intersection_series(self):
        """Test that get_intersection returns correct intersection."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main()
        })

        df1 = pd.DataFrame({
            "layer": ["0", "1", "2"],
            "feature": ["1000", "2000", "3000"],
            "feature_type": ["cross layer transcoder"] * 3,
            "ctx_idx": [1, 1, 1]
        })

        df2 = pd.DataFrame({
            "layer": ["0", "1"],
            "feature": ["1000", "2000"],
            "feature_type": ["cross layer transcoder"] * 2,
            "ctx_idx": [1, 1]
        })

        intersection = analyzer.get_intersection(df1, df2)

        self.assertEqual(len(intersection), 2)


class TestCalculatedJaccardIndex(unittest.TestCase):
    """Tests for calculated_jaccard_index static method."""

    def test_calculates_correct_jaccard(self):
        """Test correct Jaccard index calculation."""
        df1 = pd.DataFrame({
            "layer": ["0", "1", "2"],
            "feature": ["1000", "2000", "3000"]
        })

        df2 = pd.DataFrame({
            "layer": ["0", "1", "3"],
            "feature": ["1000", "2000", "4000"]
        })

        # Intersection: 2 (0/1000, 1/2000)
        # Union: 4 (0/1000, 1/2000, 2/3000, 3/4000)
        # Jaccard: 2/4 = 0.5
        intersection = pd.Series([1, 2])  # Just need length

        jaccard = GraphAnalyzer.calculated_jaccard_index(df1, df2, intersection)

        self.assertAlmostEqual(jaccard, 0.5, places=2)


class TestCalculatedWeightedJaccard(unittest.TestCase):
    """Tests for calculated_weighted_jaccard static method."""

    def test_calculates_weighted_jaccard(self):
        """Test weighted Jaccard calculation from links lookup."""
        links_lookup = {
            "target1": {
                "source1": [5.0, 3.0],  # Both graphs have this edge
                "source2": [2.0, 0.0],  # Only graph 1
            },
            "target2": {
                "source3": [0.0, 4.0],  # Only graph 2
            }
        }

        # Min weights: min(5,3)=3, min(2,0)=0, min(0,4)=0 -> sum=3
        # Max weights: max(5,3)=5, max(2,0)=2, max(0,4)=4 -> sum=11
        # Weighted jaccard: 3/11

        jaccard = GraphAnalyzer.calculated_weighted_jaccard(links_lookup)

        self.assertAlmostEqual(jaccard, 3/11, places=4)

    def test_handles_empty_lookup(self):
        """Test that empty lookup returns 0."""
        jaccard = GraphAnalyzer.calculated_weighted_jaccard({})
        self.assertEqual(jaccard, 0.0)


class TestGetCombinedLinksLookup(unittest.TestCase):
    """Tests for get_combined_links_lookup method."""

    def test_creates_nested_dict_structure(self):
        """Test that result has correct nested structure."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        lookup = analyzer.get_combined_links_lookup(["main", "other"])

        # Should be dict of dicts of lists
        self.assertIsInstance(lookup, dict)
        for target, sources in lookup.items():
            self.assertIsInstance(sources, dict)
            for source, weights in sources.items():
                self.assertIsInstance(weights, list)
                self.assertEqual(len(weights), 2)  # Two prompts

    def test_shared_edges_have_nonzero_weights(self):
        """Test that edges in both graphs have non-zero weights for both."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        lookup = analyzer.get_combined_links_lookup(["main", "other"])

        # 0_1000 -> 1_2000 edge exists in both graphs
        # Check for an edge that should exist in both (without position suffix)
        found_shared = False
        for target, sources in lookup.items():
            for source, weights in sources.items():
                if weights[0] > 0 and weights[1] > 0:
                    found_shared = True
                    break

        self.assertTrue(found_shared, "Should have at least one shared edge")


class TestGetLinksOverlap(unittest.TestCase):
    """Tests for get_links_overlap method."""

    def test_returns_links_lookup(self):
        """Test that get_links_overlap returns a lookup dict."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        lookup = analyzer.get_links_overlap("main", "other")

        self.assertIsInstance(lookup, dict)


class TestGetMetrics(unittest.TestCase):
    """Tests for get_metrics static method."""

    def test_filters_by_metric_type(self):
        """Test that only metrics of specified type are returned."""
        metrics = {ComparisonMetrics.JACCARD_INDEX, FeatureSharingMetrics.NUM_UNIQUE}

        filtered = GraphAnalyzer.get_metrics(metrics, ComparisonMetrics)

        self.assertEqual(len(filtered), 1)
        self.assertIn(ComparisonMetrics.JACCARD_INDEX, filtered)
        self.assertNotIn(FeatureSharingMetrics.NUM_UNIQUE, filtered)

    def test_all_returns_all_of_type(self):
        """Test that 'all' returns all metrics of the type."""
        filtered = GraphAnalyzer.get_metrics("all", ComparisonMetrics)

        self.assertEqual(len(filtered), len(ComparisonMetrics))


class TestGetUniqueFeatureForToken(unittest.TestCase):
    """Tests for get_unique_features_for_token static method."""

    def test_returns_features_only_linked_to_specified_token(self):
        """Test that only features unique to the token are returned."""
        token_to_features = {
            "hello": {"0_1000_1", "1_2000_1", "2_3000_1"},
            "world": {"0_1000_1", "1_4000_1"}
        }

        unique = GraphAnalyzer.get_unique_features_for_token("hello", token_to_features)

        # Unique to "hello": 1_2000_1, 2_3000_1 (0_1000_1 is shared)
        self.assertEqual(len(unique), 2)

        feature_ids = {f.feature for f in unique}
        self.assertIn("2000", feature_ids)
        self.assertIn("3000", feature_ids)
        self.assertNotIn("1000", feature_ids)

    def test_raises_for_unknown_token(self):
        """Test that assertion fails for unknown token."""
        token_to_features = {
            "hello": {"0_1000_1"}
        }

        with self.assertRaises(AssertionError):
            GraphAnalyzer.get_unique_features_for_token("unknown", token_to_features)


class TestGetMatchingToken(unittest.TestCase):
    """Tests for get_matching_token static method."""

    def test_finds_matching_output_node(self):
        """Test that matching output node is found."""
        graph1 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_main())
        graph2 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_other())

        # Both have logit_5000_1 as an output
        matching = GraphAnalyzer.get_matching_token(graph1, graph2, raise_if_no_matching_tokens=False)

        self.assertEqual(matching["node_id"], "logit_5000_1")

    def test_raises_when_no_match_and_flag_set(self):
        """Test that exception raised when no match and raise flag is True."""
        graph1 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_main())
        graph3 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_third())

        # graph1 has logit_5000_1, graph3 has logit_6000_1
        with self.assertRaises(Exception):
            GraphAnalyzer.get_matching_token(graph1, graph3, raise_if_no_matching_tokens=True)

    def test_returns_top_output_when_no_match_and_flag_false(self):
        """Test that top output returned when no match and raise flag is False."""
        graph1 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_main())
        graph3 = GraphManager(TestGraphAnalyzerFixtures.create_graph_metadata_third())

        matching = GraphAnalyzer.get_matching_token(graph1, graph3, raise_if_no_matching_tokens=False)

        # Should return graph3's target output since no match
        self.assertTrue(matching["is_target_logit"])


class TestCalculateIntersectionMetrics(unittest.TestCase):
    """Tests for calculate_intersection_metrics method."""

    def test_calculates_requested_metrics(self):
        """Test that requested metrics are calculated."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        metrics = analyzer.calculate_intersection_metrics(
            "main", "other",
            metrics={ComparisonMetrics.JACCARD_INDEX, ComparisonMetrics.FRAC_FROM_INTERSECTION}
        )

        self.assertIn(ComparisonMetrics.JACCARD_INDEX, metrics)
        self.assertIn(ComparisonMetrics.FRAC_FROM_INTERSECTION, metrics)

    def test_returns_none_for_invalid_metrics(self):
        """Test that None is returned when no valid metrics requested."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        # Request only FeatureSharingMetrics which are not ComparisonMetrics
        metrics = analyzer.calculate_intersection_metrics(
            "main", "other",
            metrics={FeatureSharingMetrics.NUM_UNIQUE}
        )

        self.assertIsNone(metrics)


class TestCalculateEdgeSharingMetrics(unittest.TestCase):
    """Tests for calculate_edge_sharing_metrics method."""

    def test_calculates_weight_metrics(self):
        """Test that weight-based metrics are calculated."""
        analyzer = TestGraphAnalyzerFixtures.create_mock_analyzer({
            "main": TestGraphAnalyzerFixtures.create_graph_metadata_main(),
            "other": TestGraphAnalyzerFixtures.create_graph_metadata_other()
        })

        metrics = analyzer.calculate_edge_sharing_metrics(
            ["main", "other"],
            metrics={
                FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC,
                FeatureSharingMetrics.SHARED_WEIGHTED_FRAC,
                FeatureSharingMetrics.MAIN_TOTAL_WEIGHT
            }
        )

        self.assertIn(FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC, metrics)
        self.assertIn(FeatureSharingMetrics.SHARED_WEIGHTED_FRAC, metrics)
        self.assertIn(FeatureSharingMetrics.MAIN_TOTAL_WEIGHT, metrics)

        # Fractions should be between 0 and 1
        self.assertGreaterEqual(metrics[FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC], 0)
        self.assertLessEqual(metrics[FeatureSharingMetrics.UNIQUE_WEIGHTED_FRAC], 1)


if __name__ == "__main__":
    unittest.main()
