import abc
from typing import Dict, List

from src.graph_manager import GraphManager


class ConfigAnalyzeStep(abc.ABC):

    def __init__(self,  **kwargs):
        """
        Runs an analysis on the graphs.

        Args:
            **kwargs: Params needed for step.
        """
        self.kwargs = kwargs

    @abc.abstractmethod
    def run(self, graphs: List[GraphManager], conditions: List[str] = None) -> Dict:
        """
        Runs the step logic.

        Args:
            graphs: Graphs to run analysis on.
            conditions: Conditions corresponding with each graph.

        Returns:
            Dictionary of results.
        """
