import unittest
import numpy as np
import pandas as pd

from src.analysis.significance_tester import (
    SignificanceTester, PairwiseResult, SIGNIFICANCE_THRESHOLD
)
from src.metrics import SignificanceMetrics


class TestPairwiseResult(unittest.TestCase):
    """Tests for PairwiseResult dataclass."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields with correct keys."""
        result = PairwiseResult(
            group1="g1", group2="g2",
            group1_mean=1.0, group2_mean=2.0,
            group1_std=0.1, group2_std=0.2,
            group1_n=10, group2_n=10,
            t_statistic=-5.0, t_p_value=0.001, t_significant=True,
            mw_u_statistic=20.0, mw_p_value=0.002, mw_significant=True,
            cohens_d=-0.8, rank_biserial_r=-0.6
        )

        d = result.to_dict()

        self.assertEqual(d['comparison'], "g1 vs g2")
        self.assertEqual(d[SignificanceMetrics.GROUP1_MEAN.value], 1.0)
        self.assertEqual(d[SignificanceMetrics.GROUP2_MEAN.value], 2.0)
        self.assertEqual(d[SignificanceMetrics.T_STATISTIC.value], -5.0)
        self.assertEqual(d[SignificanceMetrics.COHENS_D.value], -0.8)

    def test_to_dict_handles_none_values(self):
        """Test that to_dict handles None values for optional fields."""
        result = PairwiseResult(
            group1="g1", group2="g2",
            group1_mean=1.0, group2_mean=2.0,
            group1_std=0.1, group2_std=0.2,
            group1_n=10, group2_n=10
        )

        d = result.to_dict()

        self.assertIsNone(d[SignificanceMetrics.T_STATISTIC.value])
        self.assertIsNone(d[SignificanceMetrics.COHENS_D.value])


class TestSignificanceTester(unittest.TestCase):
    """Tests for SignificanceTester class."""

    def setUp(self):
        self.tester = SignificanceTester()

    def test_init_default_alpha(self):
        """Test default alpha value."""
        self.assertEqual(self.tester.alpha, SIGNIFICANCE_THRESHOLD)
        self.assertEqual(self.tester.alpha, 0.05)

    def test_init_custom_alpha(self):
        """Test custom alpha value."""
        tester = SignificanceTester(alpha=0.01)
        self.assertEqual(tester.alpha, 0.01)

    def test_compute_stats_returns_dict_with_metrics(self):
        """Test compute_stats returns dict with SignificanceMetrics keys."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        stats = self.tester.compute_stats(group1, group2)

        self.assertIn(SignificanceMetrics.GROUP1_MEAN.value, stats)
        self.assertIn(SignificanceMetrics.GROUP2_MEAN.value, stats)
        self.assertIn(SignificanceMetrics.T_STATISTIC.value, stats)
        self.assertIn(SignificanceMetrics.MW_P_VALUE.value, stats)
        self.assertIn(SignificanceMetrics.COHENS_D.value, stats)

    def test_compute_stats_two_sided(self):
        """Test compute_stats with two-sided hypothesis."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        stats = self.tester.compute_stats(group1, group2, alternative='two-sided')

        self.assertAlmostEqual(stats[SignificanceMetrics.GROUP1_MEAN.value], 3.0, places=4)
        self.assertAlmostEqual(stats[SignificanceMetrics.GROUP2_MEAN.value], 8.0, places=4)
        # Groups are clearly different
        self.assertLess(stats[SignificanceMetrics.T_P_VALUE.value], 0.05)

    def test_compute_stats_one_sided_greater(self):
        """Test compute_stats with one-sided greater hypothesis."""
        group1 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = self.tester.compute_stats(group1, group2, alternative='greater')

        # group1 > group2, so greater should be significant
        self.assertLess(stats[SignificanceMetrics.T_P_VALUE.value], 0.05)

    def test_compute_stats_one_sided_less(self):
        """Test compute_stats with one-sided less hypothesis."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        stats = self.tester.compute_stats(group1, group2, alternative='less')

        # group1 < group2, so less should be significant
        self.assertLess(stats[SignificanceMetrics.T_P_VALUE.value], 0.05)

    def test_compute_stats_small_sample(self):
        """Test compute_stats returns partial results for small samples."""
        group1 = np.array([1.0])
        group2 = np.array([2.0])

        stats = self.tester.compute_stats(group1, group2)

        self.assertEqual(stats[SignificanceMetrics.GROUP1_MEAN.value], 1.0)
        self.assertNotIn(SignificanceMetrics.T_STATISTIC.value, stats)

    def test_compare_two_groups_returns_pairwise_result(self):
        """Test compare_two_groups returns PairwiseResult."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = self.tester.compare_two_groups(group1, group2, "group1", "group2")

        self.assertIsInstance(result, PairwiseResult)
        self.assertEqual(result.group1, "group1")
        self.assertEqual(result.group2, "group2")
        self.assertEqual(result.group1_n, 5)
        self.assertEqual(result.group2_n, 5)

    def test_compare_multiple_groups_returns_dataframe(self):
        """Test compare_multiple_groups returns DataFrame with all pairs."""
        group_data = {
            "A": np.array([1.0, 2.0, 3.0]),
            "B": np.array([4.0, 5.0, 6.0]),
            "C": np.array([7.0, 8.0, 9.0])
        }

        df = self.tester.compare_multiple_groups(group_data, metric_name="test_metric")

        # Should have 3 pairs: A vs B, A vs C, B vs C
        self.assertEqual(len(df), 3)
        self.assertIn('metric', df.columns)
        self.assertIn('comparison', df.columns)
        self.assertEqual(df['metric'].iloc[0], "test_metric")

    def test_compare_multiple_groups_empty_returns_empty_df(self):
        """Test compare_multiple_groups returns empty DataFrame for single group."""
        group_data = {"A": np.array([1.0, 2.0, 3.0])}

        df = self.tester.compare_multiple_groups(group_data)

        self.assertTrue(df.empty)

    def test_get_descriptive_stats(self):
        """Test get_descriptive_stats returns per-group statistics."""
        group_data = {
            "A": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "B": np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        }

        stats = self.tester.get_descriptive_stats(group_data)

        self.assertAlmostEqual(stats['A_mean'], 3.0, places=4)
        self.assertAlmostEqual(stats['B_mean'], 8.0, places=4)
        self.assertEqual(stats['A_n'], 5)
        self.assertEqual(stats['B_n'], 5)

    def test_cohens_d_same_groups(self):
        """Test Cohen's d is 0 for identical groups."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        d = SignificanceTester.cohens_d(group, group)

        self.assertAlmostEqual(d, 0.0, places=4)

    def test_cohens_d_different_groups(self):
        """Test Cohen's d for different groups."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([4.0, 5.0, 6.0, 7.0, 8.0])

        d = SignificanceTester.cohens_d(group1, group2)

        # Group2 mean is higher, so d should be negative
        self.assertLess(d, 0)

    def test_cohens_d_zero_variance(self):
        """Test Cohen's d returns 0 when variance is 0."""
        group1 = np.array([5.0, 5.0, 5.0])
        group2 = np.array([5.0, 5.0, 5.0])

        d = SignificanceTester.cohens_d(group1, group2)

        self.assertEqual(d, 0.0)

    def test_rank_biserial_neutral(self):
        """Test rank-biserial is 0 when U = n1*n2/2."""
        n1, n2 = 10, 10
        mw_stat = (n1 * n2) / 2

        r = SignificanceTester.rank_biserial(mw_stat, n1, n2)

        self.assertAlmostEqual(r, 0.0, places=4)

    def test_rank_biserial_extremes(self):
        """Test rank-biserial at extremes."""
        n1, n2 = 10, 10

        r_max = SignificanceTester.rank_biserial(n1 * n2, n1, n2)
        self.assertAlmostEqual(r_max, 1.0, places=4)

        r_min = SignificanceTester.rank_biserial(0, n1, n2)
        self.assertAlmostEqual(r_min, -1.0, places=4)

    def test_apply_bh_correction_adds_columns(self):
        """Test apply_bh_correction adds BH-corrected columns."""
        df = pd.DataFrame({
            "metric": ["m1", "m2", "m3"],
            "t_p_value": [0.01, 0.03, 0.04],
            "mw_p_value": [0.02, 0.04, 0.06]
        })

        corrected = SignificanceTester.apply_bh_correction(df)

        self.assertIn("t_p_value_bh", corrected.columns)
        self.assertIn("mw_p_value_bh", corrected.columns)
        self.assertIn("t_significant_bh", corrected.columns)
        self.assertIn("mw_significant_bh", corrected.columns)

    def test_apply_bh_correction_handles_nan(self):
        """Test apply_bh_correction handles NaN p-values."""
        df = pd.DataFrame({
            "metric": ["m1", "m2"],
            "t_p_value": [0.01, np.nan]
        })

        corrected = SignificanceTester.apply_bh_correction(df)

        self.assertFalse(np.isnan(corrected["t_p_value_bh"].iloc[0]))
        self.assertTrue(np.isnan(corrected["t_p_value_bh"].iloc[1]))

    def test_is_significant_scalar(self):
        """Test is_significant with scalar values."""
        self.assertTrue(SignificanceTester.is_significant(0.01))
        self.assertTrue(SignificanceTester.is_significant(0.049))
        self.assertFalse(SignificanceTester.is_significant(0.05))
        self.assertFalse(SignificanceTester.is_significant(0.1))

    def test_is_significant_array(self):
        """Test is_significant with array values."""
        p_values = np.array([0.01, 0.05, 0.1])

        result = SignificanceTester.is_significant(p_values)

        np.testing.assert_array_equal(result, [True, False, False])

    def test_is_significant_custom_alpha(self):
        """Test is_significant with custom alpha."""
        self.assertTrue(SignificanceTester.is_significant(0.005, alpha=0.01))
        self.assertFalse(SignificanceTester.is_significant(0.02, alpha=0.01))


if __name__ == "__main__":
    unittest.main()
