from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import SignificanceMetrics


SIGNIFICANCE_THRESHOLD = 0.05


@dataclass
class PairwiseResult:
    """Result of a pairwise significance test between two groups."""
    group1: str
    group2: str
    group1_mean: float
    group2_mean: float
    group1_std: float
    group2_std: float
    group1_n: int
    group2_n: int
    t_statistic: Optional[float] = None
    t_p_value: Optional[float] = None
    t_significant: Optional[bool] = None
    mw_u_statistic: Optional[float] = None
    mw_p_value: Optional[float] = None
    mw_significant: Optional[bool] = None
    cohens_d: Optional[float] = None
    rank_biserial_r: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary using SignificanceMetrics enum values."""
        return {
            'comparison': f'{self.group1} vs {self.group2}',
            SignificanceMetrics.GROUP1_MEAN.value: self.group1_mean,
            SignificanceMetrics.GROUP2_MEAN.value: self.group2_mean,
            SignificanceMetrics.GROUP1_STD.value: self.group1_std,
            SignificanceMetrics.GROUP2_STD.value: self.group2_std,
            SignificanceMetrics.T_STATISTIC.value: self.t_statistic,
            SignificanceMetrics.T_P_VALUE.value: self.t_p_value,
            SignificanceMetrics.T_SIGNIFICANT.value: self.t_significant,
            SignificanceMetrics.MANN_WHITNEY_U.value: self.mw_u_statistic,
            SignificanceMetrics.MW_P_VALUE.value: self.mw_p_value,
            SignificanceMetrics.MW_SIGNIFICANT.value: self.mw_significant,
            SignificanceMetrics.COHENS_D.value: self.cohens_d,
            SignificanceMetrics.RANK_BISERIAL_R.value: self.rank_biserial_r,
        }


class SignificanceTester:
    """
    Performs statistical significance tests between groups.

    Supports:
    - Mann-Whitney U test (non-parametric)
    - Independent t-test (parametric)
    - Effect sizes (Cohen's d, rank-biserial correlation)
    - Benjamini-Hochberg FDR correction
    """

    def __init__(self, alpha: float = SIGNIFICANCE_THRESHOLD):
        """
        Args:
            alpha: Significance threshold for p-values. Defaults to 0.05.
        """
        self.alpha = alpha

    def compute_stats(self, group1: np.ndarray, group2: np.ndarray,
                      alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Compute significance test statistics between two groups.

        Args:
            group1: First group of values.
            group2: Second group of values.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary with SignificanceMetrics values as keys.
        """
        n1, n2 = len(group1), len(group2)

        result = {
            SignificanceMetrics.GROUP1_MEAN.value: float(np.mean(group1)) if n1 > 0 else np.nan,
            SignificanceMetrics.GROUP2_MEAN.value: float(np.mean(group2)) if n2 > 0 else np.nan,
            SignificanceMetrics.GROUP1_STD.value: float(np.std(group1, ddof=1)) if n1 > 1 else np.nan,
            SignificanceMetrics.GROUP2_STD.value: float(np.std(group2, ddof=1)) if n2 > 1 else np.nan,
            SignificanceMetrics.N_PER_GROUP.value: n1,
        }

        if n1 < 2 or n2 < 2:
            return result

        # T-test (parametric)
        t_stat, t_p = stats.ttest_ind(group1, group2, alternative=alternative, equal_var=False)
        result[SignificanceMetrics.T_STATISTIC.value] = float(t_stat)
        result[SignificanceMetrics.T_P_VALUE.value] = float(t_p)
        result[SignificanceMetrics.T_SIGNIFICANT.value] = t_p < self.alpha

        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p = stats.mannwhitneyu(group1, group2, alternative=alternative)
        result[SignificanceMetrics.MANN_WHITNEY_U.value] = float(mw_stat)
        result[SignificanceMetrics.MW_P_VALUE.value] = float(mw_p)
        result[SignificanceMetrics.MW_SIGNIFICANT.value] = mw_p < self.alpha

        # Effect sizes
        result[SignificanceMetrics.COHENS_D.value] = self.cohens_d(group1, group2)
        result[SignificanceMetrics.RANK_BISERIAL_R.value] = self.rank_biserial(mw_stat, n1, n2)

        return result

    def compare_two_groups(self, values1: np.ndarray, values2: np.ndarray,
                           group1_name: str = "group1",
                           group2_name: str = "group2",
                           alternative: str = 'two-sided') -> PairwiseResult:
        """
        Performs all pairwise statistical tests between two groups.

        Args:
            values1: Array of values for first group.
            values2: Array of values for second group.
            group1_name: Name of first group.
            group2_name: Name of second group.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            PairwiseResult with all test statistics.
        """
        stats_dict = self.compute_stats(values1, values2, alternative)

        return PairwiseResult(
            group1=group1_name,
            group2=group2_name,
            group1_mean=stats_dict.get(SignificanceMetrics.GROUP1_MEAN.value, np.nan),
            group2_mean=stats_dict.get(SignificanceMetrics.GROUP2_MEAN.value, np.nan),
            group1_std=stats_dict.get(SignificanceMetrics.GROUP1_STD.value, np.nan),
            group2_std=stats_dict.get(SignificanceMetrics.GROUP2_STD.value, np.nan),
            group1_n=len(values1),
            group2_n=len(values2),
            t_statistic=stats_dict.get(SignificanceMetrics.T_STATISTIC.value),
            t_p_value=stats_dict.get(SignificanceMetrics.T_P_VALUE.value),
            t_significant=stats_dict.get(SignificanceMetrics.T_SIGNIFICANT.value),
            mw_u_statistic=stats_dict.get(SignificanceMetrics.MANN_WHITNEY_U.value),
            mw_p_value=stats_dict.get(SignificanceMetrics.MW_P_VALUE.value),
            mw_significant=stats_dict.get(SignificanceMetrics.MW_SIGNIFICANT.value),
            cohens_d=stats_dict.get(SignificanceMetrics.COHENS_D.value),
            rank_biserial_r=stats_dict.get(SignificanceMetrics.RANK_BISERIAL_R.value),
        )

    def compare_multiple_groups(self, group_data: Dict[str, np.ndarray],
                                 metric_name: str = "value",
                                 alternative: str = 'two-sided') -> pd.DataFrame:
        """
        Performs pairwise comparisons between all groups.

        Args:
            group_data: Dictionary mapping group names to value arrays.
            metric_name: Name of the metric being compared.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            DataFrame with significance test results for all pairs.
        """
        groups = list(group_data.keys())
        rows = []

        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                result = self.compare_two_groups(
                    group_data[group1],
                    group_data[group2],
                    group1_name=group1,
                    group2_name=group2,
                    alternative=alternative
                )
                row = result.to_dict()
                row['metric'] = metric_name
                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    @staticmethod
    def get_descriptive_stats(group_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Computes descriptive statistics for each group.

        Args:
            group_data: Dictionary mapping group names to value arrays.

        Returns:
            Dictionary with mean, std, and n for each group.
        """
        stats_dict = {}
        for group, values in group_data.items():
            stats_dict[f'{group}_mean'] = float(np.mean(values)) if len(values) > 0 else np.nan
            stats_dict[f'{group}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            stats_dict[f'{group}_n'] = len(values)
        return stats_dict

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size using pooled standard deviation.

        Args:
            group1: First group of values.
            group2: Second group of values.

        Returns:
            Cohen's d effect size.
        """
        s1 = np.std(group1, ddof=1)
        s2 = np.std(group2, ddof=1)
        pooled_std = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
        if pooled_std > 0:
            return float((np.mean(group1) - np.mean(group2)) / pooled_std)
        return 0.0

    @staticmethod
    def rank_biserial(mw_stat: float, n1: int, n2: int) -> float:
        """
        Compute rank-biserial correlation from Mann-Whitney U statistic.

        Args:
            mw_stat: Mann-Whitney U statistic.
            n1: Size of first group.
            n2: Size of second group.

        Returns:
            Rank-biserial correlation coefficient.
        """
        return (2 * mw_stat) / (n1 * n2) - 1

    @staticmethod
    def apply_bh_correction(df: pd.DataFrame, alpha: float = SIGNIFICANCE_THRESHOLD) -> pd.DataFrame:
        """
        Apply Benjamini-Hochberg FDR correction to p-values in a DataFrame.

        Adds BH-corrected p-value and significance columns for each p-value column.

        Args:
            df: DataFrame with columns ending in 'p_value'.
            alpha: Significance threshold.

        Returns:
            DataFrame with added *_bh and *_significant_bh columns.
        """
        df = df.copy()
        for col in df.columns:
            if col.endswith("p_value"):
                p_values = df[col].values
                valid_mask = ~np.isnan(p_values)
                adjusted = np.full_like(p_values, np.nan, dtype=float)

                if valid_mask.any():
                    adjusted[valid_mask] = stats.false_discovery_control(p_values[valid_mask], method='bh')

                bh_col = f"{col}_bh"
                df[bh_col] = adjusted
                sig_col = bh_col.replace("p_value", "significant")
                df[sig_col] = adjusted < alpha

        return df

    @staticmethod
    def is_significant(p_value: float | np.ndarray,
                       alpha: float = SIGNIFICANCE_THRESHOLD) -> bool | np.ndarray:
        """
        Check if p-value(s) are below significance threshold.

        Args:
            p_value: Single p-value or array of p-values.
            alpha: Threshold for significance.

        Returns:
            Boolean or array of booleans indicating significance.
        """
        return p_value < alpha
