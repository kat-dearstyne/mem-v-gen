from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_analyze_step import CrossConfigAnalyzeStep
from src.analysis.cross_config_analysis.cross_config_subgraph_filter_step import CONFIG_NAME_COL
from src.utils import save_json, load_json

L0_VALUE_COL = "l0_value"
L0_NORMALIZED_COL = "l0_normalized"
D_TRANSCODER_COL = "d_transcoder"
LAYER_COL = "layer"
PROMPT_ID_COL = "prompt_id"
RESULTS_COL = "results"


class CrossConfigL0ReplacementModelStep(CrossConfigAnalyzeStep):
    """
    Cross-config analysis step for L0 (active feature count) results.

    Aggregates L0 per-layer values across all configs and prompts, computing
    summary statistics per layer.
    """

    CONFIG_RESULTS_KEY = SupportedConfigAnalyzeStep.L0_REPLACEMENT_MODEL
    L0_RAW_FILENAME = "l0_raw_results.json"
    L0_FILENAME = "l0_per_layer.csv"
    L0_STATS_FILENAME = "l0_stats.csv"

    def __init__(self, save_path: Path | None = None, load_path: Path | None = None, **kwargs):
        """
        Initializes the cross-config L0 replacement model step.

        Args:
            save_path: Base path for saving results.
            load_path: Path to check for cached results (defaults to save_path).
        """
        super().__init__(save_path=save_path, **kwargs)
        self.load_path = load_path or save_path

    def run(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]) -> dict | None:
        """
        Aggregates L0 results across configs, using passed-in or cached results.

        If config_results contains L0 data, extracts it, saves to JSON,
        and runs aggregation. Otherwise, attempts to load from cached JSON file.

        Args:
            config_results: Dictionary mapping config names to their per-step results.
                Each config's L0 results map prompt_id to tensor of L0 per layer.

        Returns:
            Dictionary with 'df' (raw data) and 'stats' (summary statistics),
            or None if no results found.
        """

        results = self._extract_results(config_results)

        if results:
            self._save_raw_results(results)
            return self._aggregate_results(results)

        if self.save_path is None:
            return None

        # Try to load from file (previous run check point)
        for check_path in [self.load_path, self.save_path]:
            if check_path is None:
                continue
            output_path = check_path / self.L0_RAW_FILENAME
            if output_path.exists() and (results := load_json(output_path)):
                print(f"Loaded cached L0 results from: {output_path}")
                break

        if results:
            for config_name, l0_data in results.items():
                if config_name not in config_results:
                    config_results[config_name] = {}
                config_results[config_name][self.CONFIG_RESULTS_KEY] = l0_data

            return self._aggregate_results(results)

        return None

    def _extract_results(self, config_results: dict[str, dict[SupportedConfigAnalyzeStep, Any]]
                         ) -> dict[str, dict[str, Any]] | None:
        """
        Extract L0 results from config_results if present.

        Args:
            config_results: Dictionary mapping config names to their per-step results.

        Returns:
            Extracted results in format {config_name: {"results": {...}, "d_transcoder": int}}, or None.
        """
        results = {}
        for config_name, step_results in config_results.items():
            if self.CONFIG_RESULTS_KEY in step_results:
                step_data = step_results[self.CONFIG_RESULTS_KEY]
                if step_data:
                    if isinstance(step_data, dict) and RESULTS_COL in step_data:
                        results[config_name] = step_data
        return results if results else None

    def _save_raw_results(self, results: dict[str, dict[str, Any]]) -> None:
        """
        Save raw L0 results to JSON file.

        Converts tensors to lists for JSON serialization.

        Args:
            results: Dictionary mapping config names to {"results": {...}, D_TRANSCODER_COL: int}.
        """
        self.save_path.mkdir(parents=True, exist_ok=True)
        output_path = self.save_path / self.L0_RAW_FILENAME

        serializable_results = {}
        for config_name, config_data in results.items():
            prompt_results = config_data.get(RESULTS_COL, config_data)
            d_transcoder = config_data.get(D_TRANSCODER_COL)

            serializable_results[config_name] = {
                RESULTS_COL: {},
                D_TRANSCODER_COL: d_transcoder
            }
            for prompt_id, l0_tensor in prompt_results.items():
                if prompt_id in (RESULTS_COL, D_TRANSCODER_COL):
                    continue
                if isinstance(l0_tensor, torch.Tensor):
                    l0_values = l0_tensor.cpu().tolist()
                elif isinstance(l0_tensor, np.ndarray):
                    l0_values = l0_tensor.tolist()
                else:
                    l0_values = list(l0_tensor)
                serializable_results[config_name][RESULTS_COL][prompt_id] = l0_values

        save_json(serializable_results, output_path)
        print(f"L0 raw results saved to: {output_path}")

    def _aggregate_results(self, results: dict[str, dict[str, Any]]) -> dict:
        """
        Aggregate L0 results into DataFrame and compute statistics.

        Args:
            results: Dictionary mapping config names to {"results": {...}, "d_transcoder": int}.

        Returns:
            Dictionary with 'df' (raw data) and 'stats' (summary statistics).
        """
        rows = []

        for config_name, config_data in results.items():
            prompt_results = config_data.get(RESULTS_COL, config_data)
            d_transcoder = config_data.get(D_TRANSCODER_COL)

            for prompt_id, l0_tensor in prompt_results.items():
                if prompt_id in (RESULTS_COL, D_TRANSCODER_COL):
                    continue
                if isinstance(l0_tensor, torch.Tensor):
                    l0_values = l0_tensor.cpu().numpy()
                else:
                    l0_values = np.array(l0_tensor)

                for layer, l0_value in enumerate(l0_values):
                    row = {
                        CONFIG_NAME_COL: config_name,
                        PROMPT_ID_COL: prompt_id,
                        LAYER_COL: layer,
                        L0_VALUE_COL: float(l0_value),
                        D_TRANSCODER_COL: d_transcoder,
                    }
                    # Compute normalized L0 if d_transcoder is available
                    if d_transcoder:
                        row[L0_NORMALIZED_COL] = float(l0_value) / d_transcoder
                    else:
                        row[L0_NORMALIZED_COL] = None
                    rows.append(row)

        df = pd.DataFrame(rows)
        stats_df = self._compute_layer_stats(df)

        # Save aggregated results
        self._save_aggregated_results(df, stats_df)

        return {"df": df, "stats": stats_df}

    def _compute_layer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for L0 values per layer.

        Args:
            df: DataFrame with layer and l0_value columns.

        Returns:
            DataFrame with per-layer statistics.
        """
        stats = df.groupby(LAYER_COL)[L0_VALUE_COL].agg([
            ("mean", "mean"),
            ("std", "std"),
            ("median", "median"),
            ("min", "min"),
            ("max", "max"),
            ("count", "count"),
        ]).reset_index()

        return stats

    def _save_aggregated_results(self, df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
        """
        Save aggregated L0 results to disk.

        Args:
            df: Aggregated L0 data DataFrame.
            stats_df: Summary statistics DataFrame.
        """
        # Save aggregated data
        df.to_csv(self.save_path / self.L0_FILENAME, index=False)

        # Save statistics
        stats_df.to_csv(self.save_path / self.L0_STATS_FILENAME, index=False)

        print(f"Saved L0 aggregated results to: {self.save_path}")
