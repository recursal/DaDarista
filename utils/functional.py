import pathlib
from datasets import IterableDataset
import tqdm


def line_counter(file: pathlib.Path):
    """Counts the number of lines in a file.

    Args:
        file (pathlib.Path): The file to count lines from.

    Returns:
        int: The number of lines.
    """
    lines = 0
    with open(file, "rb") as f:
        for line in tqdm.tqdm(f, desc="Counting lines in file"):
            if not line.strip():
                continue
            lines += 1
    return lines


def dataset_counter(dataset: IterableDataset, target: int) -> int:
    """Counts the number of lines in a IterableDataset.

    Args:
        dataset (IterableDataset): The dataset to count lines from.

    Returns:
        int: The number of lines.
    """
    lines = 0
    for _ in tqdm.tqdm(dataset, desc="Counting lines in IterableDataset"):
        lines += 1
        if lines > target:
            return -1
    return lines


