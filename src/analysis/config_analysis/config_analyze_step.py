import abc
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph_analyzer import GraphAnalyzer


class ConfigAnalyzeStep(abc.ABC):

    def __init__(self, graph_analyzer: "GraphAnalyzer", **kwargs):
        """
        Args:
            graph_analyzer: GraphAnalyzer instance with loaded graphs.
            **kwargs: Additional params needed for step.
        """
        self.graph_analyzer = graph_analyzer
        self.kwargs = kwargs

    @abc.abstractmethod
    def run(self) -> Any:
        """
        Runs the step logic.

        Returns:
            Results of the analysis step.
        """
