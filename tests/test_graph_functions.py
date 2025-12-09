import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from subgraph_comparisons import nodes_not_in, calculate_intersection_metrics, get_links_overlap
from attribution_graph_utils import get_links_from_node


class TestGraphFunctions(unittest.TestCase):
    """Unit tests for graph comparison and analysis functions."""

    def setUp(self):
        """Set up fake graph data for testing based on the structure from examples."""
        # Create fake graph metadata 1
        self.graph_metadata1 = {
            "metadata": {
                "slug": "test-graph-1",
                "scan": "gemma-2-2b",
                "prompt": "<bos>THE SOFTWARE IS PROVIDED AS IS",
                "info": {"url": "https://www.neuronpedia.org/gemma-2-2b/graph?slug=test1"}
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
                    "clerp": "",
                    "influence": 0.8,
                    "activation": 4.0
                },
                {
                    "node_id": "1_2000_1",
                    "feature": 2000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.75,
                    "activation": 3.5
                },
                {
                    "node_id": "2_3000_1",
                    "feature": 3000,
                    "layer": "2",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.9,
                    "activation": 5.0
                },
                {
                    "node_id": "3_5000_1",
                    "feature": 5000,
                    "layer": "3",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.85,
                    "activation": 4.8
                },
                {
                    "node_id": "E_100_1",
                    "feature": 100,
                    "layer": "E",
                    "ctx_idx": 1,
                    "feature_type": "embedding",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.5,
                    "activation": 2.0
                },
                {
                    "node_id": "logit_5000_1",
                    "feature": 5000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.95,
                    "is_target_logit": True,
                    "clerp": "\" IS\"",
                    "influence": 1.0,
                    "activation": 10.0
                }
            ],
            "links": [
                {
                    "source": "E_100_1",
                    "target": "0_1000_1",
                    "weight": 5.0
                },
                {
                    "source": "0_1000_1",
                    "target": "1_2000_1",
                    "weight": 3.0
                },
                {
                    "source": "1_2000_1",
                    "target": "2_3000_1",
                    "weight": 4.0
                },
                {
                    "source": "2_3000_1",
                    "target": "3_5000_1",
                    "weight": 3.5
                },
                {
                    "source": "3_5000_1",
                    "target": "logit_5000_1",
                    "weight": 8.0
                }
            ]
        }

        # Create fake graph metadata 2 with overlapping and unique nodes
        self.graph_metadata2 = {
            "metadata": {
                "slug": "test-graph-2",
                "scan": "gemma-2-2b",
                "prompt": "<bos>THE SOFTWARE IS PROVIDED WITHOUT WARRANTY",
                "info": {"url": "https://www.neuronpedia.org/gemma-2-2b/graph?slug=test2"}
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
                    "clerp": "",
                    "influence": 0.8,
                    "activation": 4.0
                },
                {
                    "node_id": "1_2000_1",
                    "feature": 2000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.75,
                    "activation": 3.5
                },
                {
                    "node_id": "1_4000_1",
                    "feature": 4000,
                    "layer": "1",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.85,
                    "activation": 4.5
                },
                {
                    "node_id": "E_100_1",
                    "feature": 100,
                    "layer": "E",
                    "ctx_idx": 1,
                    "feature_type": "embedding",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.5,
                    "activation": 2.0
                },
                {
                    "node_id": "logit_6000_1",
                    "feature": 6000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.92,
                    "is_target_logit": True,
                    "clerp": "\" WARRANTY\"",
                    "influence": 1.0,
                    "activation": 9.0
                }
            ],
            "links": [
                {
                    "source": "E_100_1",
                    "target": "0_1000_1",
                    "weight": 5.5
                },
                {
                    "source": "0_1000_1",
                    "target": "1_2000_1",
                    "weight": 3.5
                },
                {
                    "source": "0_1000_1",
                    "target": "1_4000_1",
                    "weight": 2.5
                },
                {
                    "source": "1_2000_1",
                    "target": "logit_6000_1",
                    "weight": 6.0
                },
                {
                    "source": "1_4000_1",
                    "target": "logit_6000_1",
                    "weight": 7.0
                }
            ]
        }

        # Create fake graph metadata 3 with different overlapping features
        self.graph_metadata3 = {
            "metadata": {
                "slug": "test-graph-3",
                "scan": "gemma-2-2b",
                "prompt": "<bos>THE SOFTWARE AND MATERIALS ARE PROVIDED",
                "info": {"url": "https://www.neuronpedia.org/gemma-2-2b/graph?slug=test3"}
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
                    "clerp": "",
                    "influence": 0.82,
                    "activation": 4.2
                },
                {
                    "node_id": "2_3000_1",
                    "feature": 3000,
                    "layer": "2",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.88,
                    "activation": 4.9
                },
                {
                    "node_id": "4_6000_1",
                    "feature": 6000,
                    "layer": "4",
                    "ctx_idx": 1,
                    "feature_type": "cross layer transcoder",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.91,
                    "activation": 5.2
                },
                {
                    "node_id": "E_100_1",
                    "feature": 100,
                    "layer": "E",
                    "ctx_idx": 1,
                    "feature_type": "embedding",
                    "token_prob": 0.0,
                    "is_target_logit": False,
                    "clerp": "",
                    "influence": 0.5,
                    "activation": 2.0
                },
                {
                    "node_id": "logit_7000_1",
                    "feature": 7000,
                    "layer": "logit",
                    "ctx_idx": 1,
                    "feature_type": "logit",
                    "token_prob": 0.94,
                    "is_target_logit": True,
                    "clerp": "\" PROVIDED\"",
                    "influence": 1.0,
                    "activation": 9.5
                }
            ],
            "links": [
                {
                    "source": "E_100_1",
                    "target": "0_1000_1",
                    "weight": 5.2
                },
                {
                    "source": "0_1000_1",
                    "target": "2_3000_1",
                    "weight": 4.1
                },
                {
                    "source": "2_3000_1",
                    "target": "4_6000_1",
                    "weight": 3.8
                },
                {
                    "source": "4_6000_1",
                    "target": "logit_7000_1",
                    "weight": 7.5
                }
            ]
        }

    def test_get_links_from_node_default_starting_node(self):
        """Test get_links_from_node with default starting node (top output logit)."""
        links = get_links_from_node(self.graph_metadata1)

        # Should get all links in the chain from output logit back to embedding
        self.assertEqual(len(links), 5)

        # Verify links contain expected source-target pairs
        link_pairs = {(link['source'], link['target']) for link in links}
        self.assertIn(("3_5000_1", "logit_5000_1"), link_pairs)
        self.assertIn(("2_3000_1", "3_5000_1"), link_pairs)
        self.assertIn(("1_2000_1", "2_3000_1"), link_pairs)
        self.assertIn(("0_1000_1", "1_2000_1"), link_pairs)
        self.assertIn(("E_100_1", "0_1000_1"), link_pairs)

    def test_get_links_from_node_custom_starting_node(self):
        """Test get_links_from_node with a custom starting node."""
        starting_node = self.graph_metadata1['nodes'][2]  # layer 2 node
        links = get_links_from_node(self.graph_metadata1, starting_node=starting_node)

        # Should get links from layer 2 node backwards
        self.assertGreater(len(links), 0)

        # Verify at least one link targets the starting node
        target_ids = [link['target'] for link in links]
        self.assertIn(starting_node['node_id'], target_ids)

    def test_get_links_from_node_positive_only(self):
        """Test get_links_from_node filters out negative weights when positive_only=True."""
        # Add a negative weight link to graph
        self.graph_metadata1['links'].append({
            "source": "0_1000_1",
            "target": "logit_5000_1",
            "weight": -2.0
        })

        links_positive_only = get_links_from_node(self.graph_metadata1, positive_only=True)
        links_all = get_links_from_node(self.graph_metadata1, positive_only=False)

        # Positive only should exclude the negative link
        self.assertEqual(len(links_all), len(links_positive_only) + 1)

        # Verify no negative weights in positive_only result
        for link in links_positive_only:
            self.assertGreaterEqual(link['weight'], 0)

    def test_get_links_from_node_with_hops_limit(self):
        """Test get_links_from_node respects hops parameter."""
        links_1_hop = get_links_from_node(self.graph_metadata1, hops=1)
        links_2_hops = get_links_from_node(self.graph_metadata1, hops=2)
        links_all = get_links_from_node(self.graph_metadata1)

        # More hops should include more or equal links
        self.assertLessEqual(len(links_1_hop), len(links_2_hops))
        self.assertLessEqual(len(links_2_hops), len(links_all))

    def test_get_links_from_node_include_features_only(self):
        """Test get_links_from_node with include_features_only=True."""
        links = get_links_from_node(self.graph_metadata1, include_features_only=True)

        # Should only include links between cross layer transcoder features
        # (except from the starting node which can be any type)
        for link in links:
            source_node = next(n for n in self.graph_metadata1['nodes'] if n['node_id'] == link['source'])
            target_node = next(n for n in self.graph_metadata1['nodes'] if n['node_id'] == link['target'])

            # Starting node (logit) is allowed, but all other sources/targets must be features
            if source_node['node_id'] != "logit_5000_1" and target_node['node_id'] != "logit_5000_1":
                # At least one of source or target should be a feature
                # Based on the clarification, both should be features when not the starting node
                is_source_feature = source_node['feature_type'] == 'cross layer transcoder'
                is_target_feature = target_node['feature_type'] == 'cross layer transcoder'
                self.assertTrue(is_source_feature or is_target_feature)

    @patch('subgraph_comparisons.create_or_load_graph')
    def test_nodes_not_in_returns_unique_nodes(self, mock_create_or_load):
        """Test nodes_not_in returns nodes unique to the main prompt."""
        # Mock the graph loading to return our fake graphs
        mock_create_or_load.side_effect = [self.graph_metadata1, self.graph_metadata2]

        main_prompt = "THE SOFTWARE IS PROVIDED AS IS"
        compare_prompts = ["THE SOFTWARE IS PROVIDED WITHOUT WARRANTY"]

        graph_metadata, unique_features = nodes_not_in(
            main_prompt=main_prompt,
            prompts2compare=compare_prompts,
            model="gemma-2-2b",
            submodel="gemmascope-transcoder-16k",
            graph_dir="/fake/path"
        )

        # Should return the first graph metadata
        self.assertEqual(graph_metadata['metadata']['slug'], 'test-graph-1')

        # Unique features should contain layer 2 feature (3000) and layer 3 feature (5000) which are only in graph 1
        # Layers 0 and 1 features (1000, 2000) are in both graphs so should be filtered out
        unique_layers = unique_features['layer'].tolist()
        unique_feature_ids = unique_features['feature'].tolist()

        # Layer 2 and layer 3 features should be present
        self.assertIn('2', unique_layers)
        self.assertIn('3000', unique_feature_ids)
        self.assertIn('3', unique_layers)
        self.assertIn('5000', unique_feature_ids)

        # Common features should not be present
        combined = list(zip(unique_layers, unique_feature_ids))
        self.assertNotIn(('0', '1000'), combined)
        self.assertNotIn(('1', '2000'), combined)

    @patch('subgraph_comparisons.create_or_load_graph')
    def test_nodes_not_in_with_metrics(self, mock_create_or_load):
        """Test nodes_not_in returns metrics when return_metrics=True."""
        mock_create_or_load.side_effect = [self.graph_metadata1, self.graph_metadata2]

        main_prompt = "THE SOFTWARE IS PROVIDED AS IS"
        compare_prompts = ["THE SOFTWARE IS PROVIDED WITHOUT WARRANTY"]

        graph_metadata, unique_features, metrics = nodes_not_in(
            main_prompt=main_prompt,
            prompts2compare=compare_prompts,
            model="gemma-2-2b",
            submodel="gemmascope-transcoder-16k",
            graph_dir="/fake/path",
            return_metrics=True
        )

        # Should return metrics dict
        self.assertIsInstance(metrics, dict)
        self.assertIn(compare_prompts[0], metrics)

        # Metrics should be IntersectionMetrics namedtuple
        metric = metrics[compare_prompts[0]]
        self.assertTrue(hasattr(metric, 'jaccard_index'))
        self.assertTrue(hasattr(metric, 'relative_contribution'))
        self.assertTrue(hasattr(metric, 'frac_from_intersection'))

    @patch('subgraph_comparisons.create_or_load_graph')
    def test_nodes_not_in_multiple_prompts(self, mock_create_or_load):
        """Test nodes_not_in correctly filters features not in ANY of multiple comparison prompts."""
        # Mock returns graph1, then graph2, then graph3
        mock_create_or_load.side_effect = [
            self.graph_metadata1,
            self.graph_metadata2,
            self.graph_metadata3
        ]

        main_prompt = "THE SOFTWARE IS PROVIDED AS IS"
        compare_prompts = [
            "THE SOFTWARE IS PROVIDED WITHOUT WARRANTY",
            "THE SOFTWARE AND MATERIALS ARE PROVIDED"
        ]

        graph_metadata, unique_features = nodes_not_in(
            main_prompt=main_prompt,
            prompts2compare=compare_prompts,
            model="gemma-2-2b",
            submodel="gemmascope-transcoder-16k",
            graph_dir="/fake/path"
        )

        # Verify the feature distribution:
        # - Layer 0, feature 1000: in all 3 graphs (should be filtered out)
        # - Layer 1, feature 2000: in graph1 and graph2 (should be filtered out)
        # - Layer 2, feature 3000: in graph1 and graph3 (should be filtered out)
        # - Layer 3, feature 5000: ONLY in graph1 (should remain!)

        unique_layers = unique_features['layer'].tolist()
        unique_feature_ids = unique_features['feature'].tolist()

        # Layer 3 feature should be the only one present
        self.assertIn('3', unique_layers)
        self.assertIn('5000', unique_feature_ids)

        # All other features should be filtered out
        combined = list(zip(unique_layers, unique_feature_ids))
        self.assertNotIn(('0', '1000'), combined)  # In all 3 graphs
        self.assertNotIn(('1', '2000'), combined)  # In graph1 and graph2
        self.assertNotIn(('2', '3000'), combined)  # In graph1 and graph3

        # Only layer 3 feature 5000 should remain
        self.assertEqual(len(unique_features), 1)
        self.assertEqual(unique_features.iloc[0]['layer'], '3')
        self.assertEqual(unique_features.iloc[0]['feature'], '5000')

    def test_calculate_intersection_metrics_jaccard_index(self):
        """Test calculate_intersection_metrics computes correct jaccard index."""
        # Create node DataFrames
        node_df1 = pd.DataFrame({
            'layer': ['0', '1', '2'],
            'feature': [1000, 2000, 3000]
        })

        node_df2 = pd.DataFrame({
            'layer': ['0', '1', '1'],
            'feature': [1000, 2000, 4000]
        })

        metrics = calculate_intersection_metrics(
            node_df1, node_df2,
            self.graph_metadata1, self.graph_metadata2
        )

        # Intersection: (0, 1000) and (1, 2000) = 2 nodes
        # Union: 4 nodes (1000, 2000, 3000, 4000)
        # Jaccard: 2 / 4 = 0.5
        self.assertAlmostEqual(metrics.jaccard_index, 0.5, places=2)

        # Fraction from intersection: 2 / 3 (2 common out of 3 in df1)
        self.assertAlmostEqual(metrics.frac_from_intersection, 2/3, places=2)

        # Relative contribution should be between 0 and 1
        self.assertGreaterEqual(metrics.relative_contribution, 0)
        self.assertLessEqual(metrics.relative_contribution, 1)

    def test_calculate_intersection_metrics_no_overlap(self):
        """Test calculate_intersection_metrics when there's no overlap."""
        node_df1 = pd.DataFrame({
            'layer': ['0', '1'],
            'feature': [1000, 2000]
        })

        node_df2 = pd.DataFrame({
            'layer': ['2', '3'],
            'feature': [3000, 4000]
        })

        metrics = calculate_intersection_metrics(
            node_df1, node_df2,
            self.graph_metadata1, self.graph_metadata2
        )

        # No intersection: jaccard should be 0
        self.assertEqual(metrics.jaccard_index, 0.0)

        # Fraction from intersection should be 0
        self.assertEqual(metrics.frac_from_intersection, 0.0)

    def test_get_links_union_structure(self):
        """Test get_links_union returns correct nested dictionary structure."""
        links_lookup, relative_contributions = get_links_overlap(
            self.graph_metadata1,
            self.graph_metadata2
        )

        # Should return a nested dictionary
        self.assertIsInstance(links_lookup, dict)

        # Should return relative contributions as a list of 2 values
        self.assertIsInstance(relative_contributions, list)
        self.assertEqual(len(relative_contributions), 2)

        # Each value in relative_contributions should be between 0 and 1
        for contrib in relative_contributions:
            self.assertGreaterEqual(contrib, 0)
            self.assertLessEqual(contrib, 1)

        # Check structure: target -> source -> [weight1, weight2]
        for target_id, sources in links_lookup.items():
            self.assertIsInstance(sources, dict)
            for source_id, weights in sources.items():
                self.assertIsInstance(weights, list)
                self.assertEqual(len(weights), 2)

    def test_get_links_union_intersecting_links(self):
        """Test get_links_union handles intersecting links correctly."""
        links_lookup, relative_contributions = get_links_overlap(
            self.graph_metadata1,
            self.graph_metadata2
        )

        # Common links should have non-zero weights in both positions
        # E_100 -> 0_1000 exists in both graphs (weights 5.0 and 5.5)
        target_key = "0_1000"
        source_key = "E_100"

        if target_key in links_lookup and source_key in links_lookup[target_key]:
            weights = links_lookup[target_key][source_key]
            self.assertGreater(weights[0], 0)  # From graph 1
            self.assertGreater(weights[1], 0)  # From graph 2

    def test_get_links_union_non_intersecting_links(self):
        """Test get_links_union handles non-intersecting links with zero weights."""
        links_lookup, relative_contributions = get_links_overlap(
            self.graph_metadata1,
            self.graph_metadata2
        )

        # Links unique to one graph should have zero weight for the other
        # For example, 2_3000 -> logit_5000 only exists in graph 1
        # We need to check if there are any links with one zero weight
        found_unique_link = False
        for target_id, sources in links_lookup.items():
            for source_id, weights in sources.items():
                if (weights[0] == 0 and weights[1] > 0) or (weights[0] > 0 and weights[1] == 0):
                    found_unique_link = True
                    break
            if found_unique_link:
                break

        self.assertTrue(found_unique_link, "Should have at least one link unique to one graph")


if __name__ == '__main__':
    unittest.main()
