import unittest
import pandas as pd
from src.graph_manager import GraphManager, Feature


class TestGraphManagerFixtures:
    """Shared test fixtures for GraphManager tests."""

    @staticmethod
    def create_simple_graph_metadata():
        """Create a simple graph with a linear chain of features."""
        return {
            "metadata": {
                "slug": "test-simple",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Test prompt here",
                "info": {"url": "https://example.com/graph/simple"}
            },
            "nodes": [
                {
                    "node_id": "E_100_1",
                    "feature": 100,
                    "layer": "E",
                    "ctx_idx": 1,
                    "feature_type": "embedding",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "embedding"
                },
                {
                    "node_id": "0_1000_1",
                    "feature": 1000,
                    "layer": "0",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "feature 1000"
                },
                {
                    "node_id": "1_2000_1",
                    "feature": 2000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "feature 2000"
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
                },
                {
                    "node_id": "logit_6000_1",
                    "feature": 6000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.03,
                    "is_target_logit": False,
                    "clerp": "\"world\""
                }
            ],
            "links": [
                {"source": "E_100_1", "target": "0_1000_1", "weight": 5.0},
                {"source": "0_1000_1", "target": "1_2000_1", "weight": 3.0},
                {"source": "1_2000_1", "target": "logit_5000_1", "weight": 8.0},
                {"source": "0_1000_1", "target": "logit_6000_1", "weight": 2.0}
            ]
        }

    @staticmethod
    def create_graph_with_duplicate_features():
        """Create a graph where the same feature appears at multiple positions."""
        return {
            "metadata": {
                "slug": "test-duplicates",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Test with duplicates",
                "info": {"url": "https://example.com/graph/duplicates"}
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
                    "node_id": "0_1000_2",
                    "feature": 1000,
                    "layer": "0",
                    "ctx_idx": 2,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": ""
                },
                {
                    "node_id": "0_1000_3",
                    "feature": 1000,
                    "layer": "0",
                    "ctx_idx": 3,
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
                    "token_prob": 0.9,
                    "is_target_logit": True,
                    "clerp": "\"test\""
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "logit_5000_1", "weight": 2.0},
                {"source": "0_1000_2", "target": "logit_5000_1", "weight": 3.0},
                {"source": "0_1000_3", "target": "logit_5000_1", "weight": 4.0}
            ]
        }

    @staticmethod
    def create_graph_with_errors():
        """Create a graph that includes error nodes."""
        return {
            "metadata": {
                "slug": "test-errors",
                "scan": "gemma-2-2b",
                "prompt": "<bos>Test with errors",
                "info": {"url": "https://example.com/graph/errors"}
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
                    "node_id": "error_500_1",
                    "feature": 500,
                    "layer": "error",
                    "ctx_idx": 1,
                    "feature_type": "transcoder_error",
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
                    "token_prob": 0.9,
                    "is_target_logit": True,
                    "clerp": "\"result\""
                }
            ],
            "links": [
                {"source": "0_1000_1", "target": "logit_5000_1", "weight": 5.0},
                {"source": "error_500_1", "target": "logit_5000_1", "weight": 1.0}
            ]
        }


class TestGraphManagerInit(unittest.TestCase):
    """Tests for GraphManager initialization."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_init_stores_metadata(self):
        """Test that init stores the graph metadata."""
        self.assertEqual(self.graph.graph_metadata, self.metadata)

    def test_init_stores_nodes(self):
        """Test that init stores nodes from metadata."""
        self.assertEqual(self.graph.nodes, self.metadata["nodes"])
        self.assertEqual(len(self.graph.nodes), 5)

    def test_init_stores_links(self):
        """Test that init stores links from metadata."""
        self.assertEqual(self.graph.links, self.metadata["links"])
        self.assertEqual(len(self.graph.links), 4)

    def test_init_stores_prompt(self):
        """Test that init stores the prompt."""
        self.assertEqual(self.graph.prompt, "<bos>Test prompt here")

    def test_init_stores_url(self):
        """Test that init stores the URL."""
        self.assertEqual(self.graph.url, "https://example.com/graph/simple")

    def test_init_creates_node_dict(self):
        """Test that init creates a node_id to node mapping."""
        self.assertIn("0_1000_1", self.graph.node_dict)
        self.assertEqual(self.graph.node_dict["0_1000_1"]["feature"], 1000)


class TestCreateNodeDf(unittest.TestCase):
    """Tests for GraphManager.create_node_df method."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_creates_dataframe_with_expected_columns(self):
        """Test that create_node_df returns a DataFrame with correct columns."""
        df = self.graph.create_node_df()
        expected_columns = {"feature", "layer", "feature_type", "ctx_idx"}
        self.assertEqual(set(df.columns), expected_columns)

    def test_includes_all_node_types_by_default(self):
        """Test that all node types are included by default."""
        df = self.graph.create_node_df()
        feature_types = set(df["feature_type"].tolist())
        self.assertIn("embedding", feature_types)
        self.assertIn("cross layer transcoder", feature_types)
        self.assertIn("logit", feature_types)

    def test_exclude_embeddings(self):
        """Test that embeddings can be excluded."""
        df = self.graph.create_node_df(exclude_embeddings=True)
        self.assertNotIn("embedding", df["feature_type"].tolist())

    def test_exclude_logits(self):
        """Test that logits can be excluded."""
        df = self.graph.create_node_df(exclude_logits=True)
        self.assertNotIn("logit", df["feature_type"].tolist())

    def test_exclude_errors(self):
        """Test that errors can be excluded."""
        metadata = TestGraphManagerFixtures.create_graph_with_errors()
        graph = GraphManager(metadata)

        df_with_errors = graph.create_node_df(exclude_errors=False)
        df_without_errors = graph.create_node_df(exclude_errors=True)

        self.assertGreater(len(df_with_errors), len(df_without_errors))
        self.assertIn("transcoder_error", df_with_errors["feature_type"].tolist())
        self.assertNotIn("transcoder_error", df_without_errors["feature_type"].tolist())

    def test_drop_duplicates(self):
        """Test that drop_duplicates removes duplicate layer/feature combinations."""
        metadata = TestGraphManagerFixtures.create_graph_with_duplicate_features()
        graph = GraphManager(metadata)

        df_with_dupes = graph.create_node_df(drop_duplicates=False)
        df_without_dupes = graph.create_node_df(drop_duplicates=True)

        # Should have 3 instances of feature 1000 without dedup
        layer_0_count = len(df_with_dupes[df_with_dupes["feature"] == "1000"])
        self.assertEqual(layer_0_count, 3)

        # Should have 1 instance with dedup
        layer_0_count_deduped = len(df_without_dupes[df_without_dupes["feature"] == "1000"])
        self.assertEqual(layer_0_count_deduped, 1)


class TestGetFrequencies(unittest.TestCase):
    """Tests for frequency calculation methods."""

    def test_get_frequencies_counts_cross_layer_transcoders(self):
        """Test that get_frequencies counts only cross layer transcoder features."""
        metadata = TestGraphManagerFixtures.create_graph_with_duplicate_features()
        graph = GraphManager(metadata)
        df = graph.create_node_df()

        freq_df = GraphManager.get_frequencies(df)

        # Should have one entry for the repeated feature
        self.assertEqual(len(freq_df), 1)
        self.assertEqual(freq_df.iloc[0]["ctx_freq"], 3)

    def test_get_frequencies_from_graph(self):
        """Test get_frequencies_from_graph creates df if not provided."""
        metadata = TestGraphManagerFixtures.create_graph_with_duplicate_features()
        graph = GraphManager(metadata)

        freq_df = graph.get_frequencies_from_graph()

        self.assertIn("ctx_freq", freq_df.columns)
        self.assertGreater(len(freq_df), 0)


class TestNodeIdParsing(unittest.TestCase):
    """Tests for node ID parsing and creation."""

    def test_get_feature_from_node_id_default_delimiter(self):
        """Test parsing node ID with default delimiter."""
        feature = GraphManager.get_feature_from_node_id("5-1000")
        self.assertEqual(feature.layer, "5")
        self.assertEqual(feature.feature, "1000")

    def test_get_feature_from_node_id_underscore_delimiter(self):
        """Test parsing node ID with underscore delimiter."""
        feature = GraphManager.get_feature_from_node_id("5_1000", deliminator="_")
        self.assertEqual(feature.layer, "5")
        self.assertEqual(feature.feature, "1000")

    def test_create_node_id_default_delimiter(self):
        """Test creating node ID with default delimiter."""
        feature = Feature("5", "1000")
        node_id = GraphManager.create_node_id(feature)
        self.assertEqual(node_id, "5-1000")

    def test_create_node_id_custom_delimiter(self):
        """Test creating node ID with custom delimiter."""
        feature = Feature("5", "1000")
        node_id = GraphManager.create_node_id(feature, deliminator="_")
        self.assertEqual(node_id, "5_1000")

    def test_get_id_without_pos_removes_position(self):
        """Test that position suffix is removed from node IDs."""
        self.assertEqual(GraphManager.get_id_without_pos("5_1000_3"), "5_1000")

    def test_get_id_without_pos_preserves_ids_without_position(self):
        """Test that IDs without position suffix are unchanged."""
        self.assertEqual(GraphManager.get_id_without_pos("5_1000"), "5_1000")
        self.assertEqual(GraphManager.get_id_without_pos("logit"), "logit")

    def test_get_node_ids_from_features(self):
        """Test creating node IDs from a DataFrame of features."""
        df = pd.DataFrame({
            "layer": ["0", "1", "2"],
            "feature": ["1000", "2000", "3000"]
        })
        node_ids = GraphManager.get_node_ids_from_features(df)
        self.assertEqual(node_ids, ["0-1000", "1-2000", "2-3000"])


class TestOutputTokenExtraction(unittest.TestCase):
    """Tests for output token extraction from clerp."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_get_output_token_from_clerp_extracts_token(self):
        """Test extracting token from clerp attribute."""
        output_node = self.graph.get_top_output_logit_node()
        token = self.graph.get_output_token_from_clerp(output_node)
        self.assertEqual(token, "hello")

    def test_get_output_token_from_clerp_default_node(self):
        """Test that default node is used if none provided."""
        token = self.graph.get_output_token_from_clerp()
        self.assertEqual(token, "hello")

    def test_get_output_token_handles_quote_token(self):
        """Test that a quote character as token is handled correctly."""
        # Modify a node to have clerp with just quotes
        self.graph.nodes[3]["clerp"] = '"\"\""'
        token = self.graph.get_output_token_from_clerp(self.graph.nodes[3])
        self.assertEqual(token, '"')


class TestOutputLogitNodes(unittest.TestCase):
    """Tests for output logit node methods."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_get_top_output_logit_node_returns_target(self):
        """Test that get_top_output_logit_node returns the target logit."""
        node = self.graph.get_top_output_logit_node()
        self.assertTrue(node["is_target_logit"])
        self.assertEqual(node["node_id"], "logit_5000_1")

    def test_get_output_logits_returns_all_logits(self):
        """Test that get_output_logits returns all logit nodes."""
        logits = self.graph.get_output_logits()
        self.assertEqual(len(logits), 2)
        for logit in logits:
            self.assertEqual(logit["feature_type"], "logit")

    def test_find_output_node_finds_matching_node(self):
        """Test that find_output_node finds a matching node by ID prefix."""
        node_to_find = {"node_id": "logit_5000_1", "clerp": "test"}
        found = self.graph.find_output_node(node_to_find)
        self.assertEqual(found["node_id"], "logit_5000_1")

    def test_find_output_node_returns_none_when_not_found(self):
        """Test that find_output_node returns None when no match found."""
        node_to_find = {"node_id": "logit_9999_1", "clerp": "nonexistent"}
        found = self.graph.find_output_node(node_to_find)
        self.assertIsNone(found)

    def test_find_output_node_raises_when_not_found_and_flag_set(self):
        """Test that find_output_node raises when no match and raise_if_not_found=True."""
        node_to_find = {"node_id": "logit_9999_1", "clerp": "nonexistent"}
        with self.assertRaises(AssertionError):
            self.graph.find_output_node(node_to_find, raise_if_not_found=True)


class TestLinksFromNode(unittest.TestCase):
    """Tests for get_links_from_node method."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_get_links_from_node_default_starts_from_target(self):
        """Test that default starting node is the target logit."""
        links = self.graph.get_links_from_node()
        # Should find all links in the chain
        self.assertGreater(len(links), 0)

    def test_get_links_from_node_respects_hops(self):
        """Test that hops parameter limits traversal depth."""
        links_1_hop = self.graph.get_links_from_node(hops=1)
        links_all = self.graph.get_links_from_node()

        self.assertLessEqual(len(links_1_hop), len(links_all))

    def test_get_links_from_node_positive_only(self):
        """Test that positive_only filters negative weights."""
        # Add a negative weight link
        self.graph.links.append({
            "source": "E_100_1",
            "target": "logit_5000_1",
            "weight": -1.0
        })

        links_positive = self.graph.get_links_from_node(positive_only=True)
        links_all = self.graph.get_links_from_node(positive_only=False)

        self.assertLess(len(links_positive), len(links_all))

    def test_get_links_from_node_include_features_only(self):
        """Test that include_features_only filters to cross layer transcoders."""
        links = self.graph.get_links_from_node(include_features_only=True)

        # All source nodes should be cross layer transcoders
        for link in links:
            source_node = self.graph.node_dict.get(link["source"])
            if source_node:
                self.assertEqual(source_node["feature_type"], "cross layer transcoder")


class TestCheckFeaturePresence(unittest.TestCase):
    """Tests for check_feature_presence method."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_returns_true_for_existing_feature(self):
        """Test that existing feature is found."""
        self.assertTrue(self.graph.check_feature_presence("0", "1000"))

    def test_returns_false_for_nonexistent_feature(self):
        """Test that nonexistent feature is not found."""
        self.assertFalse(self.graph.check_feature_presence("99", "9999"))


class TestGetNodesLinkedToTarget(unittest.TestCase):
    """Tests for get_nodes_linked_to_target method."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_returns_linked_nodes(self):
        """Test that nodes linked to target are returned."""
        target = self.graph.get_top_output_logit_node()
        linked = self.graph.get_nodes_linked_to_target(target)

        self.assertGreater(len(linked), 0)

    def test_sorts_by_weight_when_requested(self):
        """Test that results are sorted by weight descending."""
        target = self.graph.get_top_output_logit_node()
        linked = self.graph.get_nodes_linked_to_target(target, should_sort=True)

        # Should be sorted - first node should have highest weight link
        if len(linked) > 1:
            # Get weights for the links
            links = self.graph.get_links_from_node(target, hops=1)
            link_weights = {link["source"]: link["weight"] for link in links}
            sorted_weights = sorted(link_weights.values(), reverse=True)
            # First linked node should have the highest weight
            first_node_weight = link_weights.get(linked[0]["node_id"], 0)
            self.assertEqual(first_node_weight, sorted_weights[0])


class TestGetFeaturesLinkedToTokens(unittest.TestCase):
    """Tests for get_features_linked_to_tokens method."""

    def setUp(self):
        self.metadata = TestGraphManagerFixtures.create_simple_graph_metadata()
        self.graph = GraphManager(self.metadata)

    def test_returns_dict_of_token_to_features(self):
        """Test that method returns a dict mapping tokens to feature sets."""
        result = self.graph.get_features_linked_to_tokens(2)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two logit nodes

    def test_includes_all_linked_features(self):
        """Test that all transitively linked features are included."""
        result = self.graph.get_features_linked_to_tokens(1)

        # The target token should have features linked through the chain
        token = "hello"
        self.assertIn(token, result)
        # Should include the logit node itself plus linked features
        self.assertGreater(len(result[token]), 1)


if __name__ == "__main__":
    unittest.main()
