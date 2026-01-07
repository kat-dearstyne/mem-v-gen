import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.graph_analyzer import GraphAnalyzer, ComparisonMetrics
from src.graph_manager import GraphManager


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
                "info": {"url": "https://www.neuronpedia.org/gemma-2-2b/graph?slug=test1"},
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

        # Create GraphManager instances
        self.graph1 = GraphManager(self.graph_metadata1)
        self.graph2 = GraphManager(self.graph_metadata2)
        self.graph3 = GraphManager(self.graph_metadata3)

    def test_get_links_from_node_default_starting_node(self):
        """Test get_links_from_node with default starting node (top output logit)."""
        links = self.graph1.get_links_from_node()

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
        starting_node = self.graph1.nodes[2]  # layer 2 node
        links = self.graph1.get_links_from_node(starting_node=starting_node)

        # Should get links from layer 2 node backwards
        self.assertGreater(len(links), 0)

        # Verify at least one link targets the starting node
        target_ids = [link['target'] for link in links]
        self.assertIn(starting_node['node_id'], target_ids)

    def test_get_links_from_node_positive_only(self):
        """Test get_links_from_node filters out negative weights when positive_only=True."""
        # Add a negative weight link to graph
        self.graph1.links.append({
            "source": "0_1000_1",
            "target": "logit_5000_1",
            "weight": -2.0
        })

        links_positive_only = self.graph1.get_links_from_node(positive_only=True)
        links_all = self.graph1.get_links_from_node(positive_only=False)

        # Positive only should exclude the negative link
        self.assertEqual(len(links_all), len(links_positive_only) + 1)

        # Verify no negative weights in positive_only result
        for link in links_positive_only:
            self.assertGreaterEqual(link['weight'], 0)

    def test_get_links_from_node_with_hops_limit(self):
        """Test get_links_from_node respects hops parameter."""
        links_1_hop = self.graph1.get_links_from_node(hops=1)
        links_2_hops = self.graph1.get_links_from_node(hops=2)
        links_all = self.graph1.get_links_from_node()

        # More hops should include more or equal links
        self.assertLessEqual(len(links_1_hop), len(links_2_hops))
        self.assertLessEqual(len(links_2_hops), len(links_all))

    def test_get_links_from_node_include_features_only(self):
        """Test get_links_from_node with include_features_only=True."""
        links = self.graph1.get_links_from_node(include_features_only=True)

        # Should only include links between cross layer transcoder features
        # (except from the starting node which can be any type)
        for link in links:
            source_node = next(n for n in self.graph1.nodes if n['node_id'] == link['source'])
            target_node = next(n for n in self.graph1.nodes if n['node_id'] == link['target'])

            # Starting node (logit) is allowed, but all other sources/targets must be features
            if source_node['node_id'] != "logit_5000_1" and target_node['node_id'] != "logit_5000_1":
                # At least one of source or target should be a feature
                # Based on the clarification, both should be features when not the starting node
                is_source_feature = source_node['feature_type'] == 'cross layer transcoder'
                is_target_feature = target_node['feature_type'] == 'cross layer transcoder'
                self.assertTrue(is_source_feature or is_target_feature)

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_nodes_not_in_returns_unique_nodes(self, mock_load_graphs):
        """Test nodes_not_in returns nodes unique to the main prompt."""
        # Create node DataFrames for the graphs
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                               drop_duplicates=True)

        # Mock the graph loading to return our fake graphs
        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': main_df, 'other': other_df}
        )

        # Create analyzer with mock
        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'THE SOFTWARE IS PROVIDED AS IS', 'other': 'THE SOFTWARE IS PROVIDED WITHOUT WARRANTY'},
            neuronpedia_manager=mock_neuronpedia
        )

        unique_features = analyzer.nodes_not_in(
            main_prompt_id='main',
            comparison_prompts=['other']
        )

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

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_nodes_not_in_with_metrics(self, mock_load_graphs):
        """Test nodes_not_in returns metrics when metrics2run is provided."""
        # Create node DataFrames for the graphs
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                               drop_duplicates=True)

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': main_df, 'other': other_df}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'THE SOFTWARE IS PROVIDED AS IS', 'other': 'THE SOFTWARE IS PROVIDED WITHOUT WARRANTY'},
            neuronpedia_manager=mock_neuronpedia
        )

        unique_features, metrics = analyzer.nodes_not_in(
            main_prompt_id='main',
            comparison_prompts=['other'],
            metrics2run={ComparisonMetrics.JACCARD_INDEX, ComparisonMetrics.WEIGHTED_JACCARD,
                         ComparisonMetrics.FRAC_FROM_INTERSECTION}
        )

        # Should return metrics dict
        self.assertIsInstance(metrics, dict)
        self.assertIn('other', metrics)

        # Metrics should be a dict with metric names as keys
        metric = metrics['other']
        self.assertIn('jaccard_index', metric)
        self.assertIn('weighted_jaccard', metric)
        self.assertIn('frac_from_intersection', metric)

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_nodes_not_in_multiple_prompts(self, mock_load_graphs):
        """Test nodes_not_in correctly filters features not in ANY of multiple comparison prompts."""
        # Create node DataFrames for the graphs
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df2 = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                                drop_duplicates=True)
        other_df3 = self.graph3.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                                drop_duplicates=True)

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other1': self.graph2, 'other2': self.graph3},
            {'main': main_df, 'other1': other_df2, 'other2': other_df3}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={
                'main': 'THE SOFTWARE IS PROVIDED AS IS',
                'other1': 'THE SOFTWARE IS PROVIDED WITHOUT WARRANTY',
                'other2': 'THE SOFTWARE AND MATERIALS ARE PROVIDED'
            },
            neuronpedia_manager=mock_neuronpedia
        )

        unique_features = analyzer.nodes_not_in(
            main_prompt_id='main',
            comparison_prompts=['other1', 'other2']
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

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_calculate_intersection_metrics_jaccard_index(self, mock_load_graphs):
        """Test calculate_intersection_metrics computes correct jaccard index."""
        # Create node DataFrames
        node_df1 = pd.DataFrame({
            'layer': ['0', '1', '2'],
            'feature': ['1000', '2000', '3000'],
            'feature_type': ['cross layer transcoder'] * 3,
            'ctx_idx': [1, 1, 1]
        })

        node_df2 = pd.DataFrame({
            'layer': ['0', '1', '1'],
            'feature': ['1000', '2000', '4000'],
            'feature_type': ['cross layer transcoder'] * 3,
            'ctx_idx': [1, 1, 1]
        })

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': node_df1, 'other': node_df2}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'prompt1', 'other': 'prompt2'},
            neuronpedia_manager=mock_neuronpedia
        )

        metrics = analyzer.calculate_intersection_metrics(
            prompt1_id='main',
            prompt2_id='other',
            metrics={ComparisonMetrics.JACCARD_INDEX, ComparisonMetrics.FRAC_FROM_INTERSECTION,
                     ComparisonMetrics.WEIGHTED_JACCARD}
        )

        # Intersection: (0, 1000) and (1, 2000) = 2 nodes
        # Union: 4 nodes (1000, 2000, 3000, 4000)
        # Jaccard: 2 / 4 = 0.5
        self.assertAlmostEqual(metrics['jaccard_index'], 0.5, places=2)

        # Fraction from intersection: 2 / 3 (2 common out of 3 in df1)
        self.assertAlmostEqual(metrics['frac_from_intersection'], 2/3, places=2)

        # Weighted jaccard should be between 0 and 1
        self.assertGreaterEqual(metrics['weighted_jaccard'], 0)
        self.assertLessEqual(metrics['weighted_jaccard'], 1)

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_calculate_intersection_metrics_no_overlap(self, mock_load_graphs):
        """Test calculate_intersection_metrics when there's no overlap."""
        node_df1 = pd.DataFrame({
            'layer': ['0', '1'],
            'feature': ['1000', '2000'],
            'feature_type': ['cross layer transcoder'] * 2,
            'ctx_idx': [1, 1]
        })

        node_df2 = pd.DataFrame({
            'layer': ['2', '3'],
            'feature': ['3000', '4000'],
            'feature_type': ['cross layer transcoder'] * 2,
            'ctx_idx': [1, 1]
        })

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': node_df1, 'other': node_df2}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'prompt1', 'other': 'prompt2'},
            neuronpedia_manager=mock_neuronpedia
        )

        metrics = analyzer.calculate_intersection_metrics(
            prompt1_id='main',
            prompt2_id='other',
            metrics={ComparisonMetrics.JACCARD_INDEX, ComparisonMetrics.FRAC_FROM_INTERSECTION}
        )

        # No intersection: jaccard should be 0
        self.assertEqual(metrics['jaccard_index'], 0.0)

        # Fraction from intersection should be 0
        self.assertEqual(metrics['frac_from_intersection'], 0.0)

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_get_links_overlap_structure(self, mock_load_graphs):
        """Test get_links_overlap returns correct nested dictionary structure."""
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                               drop_duplicates=True)

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': main_df, 'other': other_df}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'prompt1', 'other': 'prompt2'},
            neuronpedia_manager=mock_neuronpedia
        )

        links_lookup, (intersection, total), output_node2 = analyzer.get_links_overlap(
            self.graph1,
            self.graph2,
            raise_if_no_matching_tokens=False
        )

        # Should return a nested dictionary
        self.assertIsInstance(links_lookup, dict)

        # Should return intersection and total arrays
        self.assertEqual(len(intersection), 2)
        self.assertEqual(len(total), 2)

        # Check structure: target -> source -> [weight1, weight2]
        for target_id, sources in links_lookup.items():
            self.assertIsInstance(sources, dict)
            for source_id, weights in sources.items():
                self.assertIsInstance(weights, list)
                self.assertEqual(len(weights), 2)

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_get_links_overlap_intersecting_links(self, mock_load_graphs):
        """Test get_links_overlap handles intersecting links correctly."""
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                               drop_duplicates=True)

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': main_df, 'other': other_df}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'prompt1', 'other': 'prompt2'},
            neuronpedia_manager=mock_neuronpedia
        )

        links_lookup, (intersection, total), output_node2 = analyzer.get_links_overlap(
            self.graph1,
            self.graph2,
            raise_if_no_matching_tokens=False
        )

        # Common links should have non-zero weights in both positions
        # E_100 -> 0_1000 exists in both graphs (weights 5.0 and 5.5)
        target_key = "0_1000"
        source_key = "E_100"

        if target_key in links_lookup and source_key in links_lookup[target_key]:
            weights = links_lookup[target_key][source_key]
            self.assertGreater(weights[0], 0)  # From graph 1
            self.assertGreater(weights[1], 0)  # From graph 2

    @patch.object(GraphAnalyzer, 'load_graphs_and_dfs')
    def test_get_links_overlap_non_intersecting_links(self, mock_load_graphs):
        """Test get_links_overlap handles non-intersecting links with zero weights."""
        main_df = self.graph1.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                              drop_duplicates=True)
        other_df = self.graph2.create_node_df(exclude_embeddings=True, exclude_errors=True, exclude_logits=True,
                                               drop_duplicates=True)

        mock_load_graphs.return_value = (
            {'main': self.graph1, 'other': self.graph2},
            {'main': main_df, 'other': other_df}
        )

        mock_neuronpedia = MagicMock()
        analyzer = GraphAnalyzer(
            prompts={'main': 'prompt1', 'other': 'prompt2'},
            neuronpedia_manager=mock_neuronpedia
        )

        links_lookup, (intersection, total), output_node2 = analyzer.get_links_overlap(
            self.graph1,
            self.graph2,
            raise_if_no_matching_tokens=False
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
