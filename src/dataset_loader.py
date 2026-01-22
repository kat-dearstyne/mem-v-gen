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
            seed: int = DEFAULT_SEED,
            max_length: int = None
    ):
        """
        Initialize the dataset loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb-edu').
            text_column: Column containing text data.
            split: Dataset split to use.
            subset: Dataset subset/config name if required (e.g., 'en' for c4).
            seed: Random seed for reproducible sampling.
            max_length: Maximum character length to truncate samples to.
        """
        self.dataset_name = dataset_name
        self.text_column = text_column
        self.split = split
        self.subset = subset
        self.seed = seed
        self.max_length = max_length
        self._dataset = None

    def load(self, num_samples: int) -> Dict[str, str]:
        """
        Load a random sample of prompts from the dataset using streaming.

        Args:
            num_samples: Number of samples to randomly select.

        Returns:
            Dictionary mapping prompt_id to prompt string.
        """
        # Use streaming to avoid downloading the entire dataset
        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True
        )

        # Shuffle with buffer and take only what we need
        shuffled = dataset.shuffle(seed=self.seed, buffer_size=10000)

        prompts = {}
        dataset_short_name = self.dataset_name.split('/')[-1]
        for i, sample in enumerate(shuffled):
            if i >= num_samples:
                break
            prompt_id = f"{dataset_short_name}_{i}"
            text = sample[self.text_column]
            if self.max_length is not None:
                text = text[:self.max_length]
            prompts[prompt_id] = text

        return prompts
