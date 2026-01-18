from typing import Dict

from datasets import load_dataset


class DatasetPromptsLoader:
    """Loads prompts from HuggingFace datasets for L0 analysis."""

    DEFAULT_SEED = 42

    def __init__(
            self,
            dataset_name: str,
            text_column: str = "text",
            split: str = "train",
            subset: str = None,
            seed: int = DEFAULT_SEED
    ):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb-edu').
            text_column: Column containing text data.
            split: Dataset split to use.
            subset: Dataset subset/config name if required (e.g., 'en' for c4).
            seed: Random seed for reproducible sampling.
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.split = split
        self.subset = subset
        self.seed = seed
        self._dataset = None

    def load(self, num_samples: int) -> Dict[str, str]:
        """
        Load a random sample of prompts from the dataset.

        Args:
            num_samples: Number of samples to randomly select.

        Returns:
            Dictionary mapping prompt_id to prompt string.
        """
        if self._dataset is None:
            self._dataset = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split
            )

        sampled = self._dataset.shuffle(seed=self.seed).select(range(num_samples))

        prompts = {}
        dataset_short_name = self.dataset_name.split('/')[-1]
        for i, sample in enumerate(sampled):
            prompt_id = f"{dataset_short_name}_{i}"
            prompts[prompt_id] = sample[self.text_column]

        # Clear dataset from memory
        del self._dataset
        del sampled
        self._dataset = None

        return prompts
