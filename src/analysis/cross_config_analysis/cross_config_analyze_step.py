import abc
from pathlib import Path
from typing import Dict, Any

from src.analysis.config_analysis.supported_config_analyze_step import SupportedConfigAnalyzeStep


class CrossConfigAnalyzeStep(abc.ABC):

    def __init__(self,  save_path: Path = None, **kwargs):
        """
        Initializes the cross-config error ranking step.

        Args:
            save_path: Base path for saving results
             **kwargs: Params needed for step.
         """
        self.kwargs = kwargs
        self.save_path = save_path

    @abc.abstractmethod
    def run(self, config_results: Dict[str, Dict[SupportedConfigAnalyzeStep, Any]]) -> Dict:
        """
        Runs the step logic.

        Args:
            config_results: Results for all configs.

        Returns:
            Dictionary of results.
        """