import logging
import mmap
import pathlib
import random

import tqdm
from .dist_by_length import DistByLength


def shuffle_mmaped(
    in_file: pathlib.Path, out_file: pathlib.Path, seed: int, logger: logging.Logger
):
    """Shuffles the file with indices using mmap

    Args:
        in_file (pathlib.Path): The jsonl file to be shuffled
        out_file (pathlib.Path): The resultant shuffled file
        seed (int): seed for the rng generator.
    """
    rng = random.Random()
    rng.seed(seed)
    # mmap seems to be available for windows and linux for python 3.5 and above.
    logger.info("Shuffling...")
    line_indices = []
    with open(in_file, "rb") as f:
        start = 0
        for _ in tqdm.tqdm(
            f,
            desc="Finding line indices",
            unit="b",
            unit_scale=True,
            dynamic_ncols=True,
        ):
            pos = f.tell()
            line_indices.append((start, pos))
            start = pos + 1
    rng.shuffle(line_indices)
    with open(in_file, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as mem_data:
        with open(out_file, "wb") as out:
            for idx in tqdm.tqdm(line_indices, desc="Indices written"):
                mem = mem_data[idx[0] : idx[1]]
                if not mem.endswith(b"\n"):
                    mem += b"\n"
                if not mem.startswith(b"{"):
                    mem = b"{" + mem
                out.write(mem)
            # print(idx)
    del rng


def dist_by_length_mmapped(
    in_file: pathlib.Path,
    out_file: pathlib.Path,
    seed: int,
    min_lines_per_group: int,
    min_subchunk_size: int,
    logger: logging.Logger,
):
    """Distributes by groups of context length
    Args:
        in_file (pathlib.Path): The jsonl file to be shuffled
        out_file (pathlib.Path): The resultant shuffled file
        seed (int): seed for the rng generator.
        min_lines_per_group (int): Minimum number of lines per group
        min_subchunk_size (int): Minimum number of lines per subchunk
    """
    logger.info("Distributing chunks by length...")
    processor = DistByLength(seed)

    line_indices = processor.get_line_indices(in_file)

    logger.info("Group lines by length...")
    groups = processor.group_lines(
        line_indices, min_lines_per_group=min_lines_per_group
    )

    logger.info("Subchunking groups...")
    groups = processor.chunk_evenly(groups, min_subchunk_size)

    # remove empty groups
    groups = [group for group in groups if group]

    logger.info("Shuffle and write groups...")
    processor.shuffle_and_write_groups(in_file, out_file, groups)
