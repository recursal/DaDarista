import mmap
import tqdm
from typing import List, Tuple
from math import gcd
import random

class DistByLength:
    def __init__(self, seed):
        self.random = random
        self.random.seed(seed)
    
    def get_line_indices(self, file: str) -> List[Tuple[int, int, int]]:
        line_indices = []
        line_count = 0
        with open(file, "rb") as f:
            start = 0
            for _ in tqdm.tqdm(
                f,
                desc="Finding line indices",
                unit="b",
                unit_scale=True,
                dynamic_ncols=True,
            ):
                pos = f.tell()
                line_length = pos - start
                line_indices.append((start, pos, line_length))
                start = pos + 1
                line_count += 1

        return line_indices
    
    def gcd_list(self, numbers):
        """
        Calculate the greatest common divisor of a list of numbers.
        """
        result = numbers[0]
        for i in range(1, len(numbers)):
            result = gcd(result, numbers[i])
        return result

    def group_lines(self, line_indices: List[Tuple[int, int, int]], min_lines_per_group: int = 1) -> List[List[Tuple[int, int, int]]]:
        """
        Groups lines into groups of approximately equal size, with the goal of minimizing the difference between group sizes.
        Each group will have at least `min_lines_per_group` lines.
        """
        total_lines = sum(line_length for _, _, line_length in line_indices)
        line_lengths = [line_length for _, _, line_length in line_indices]
        gcd_line_lengths = self.gcd_list(line_lengths)
        possible_group_sizes = [i for i in range(max(gcd_line_lengths, min_lines_per_group), total_lines // 2 + 1) if total_lines % i == 0]

        min_diff = float('inf')
        best_group_size = None

        for group_size in possible_group_sizes:
            groups = [[]]
            current_group_size = 0
            for start, end_pos, line_length in sorted(line_indices, key=lambda x: x[2], reverse=True):
                if current_group_size + line_length > group_size:
                    if len(groups[-1]) >= min_lines_per_group:
                        groups.append([])
                        current_group_size = 0
                    else:
                        continue

                groups[-1].append((start, end_pos, line_length))
                current_group_size += line_length

            if len(groups[-1]) < min_lines_per_group:
                groups = groups[:-1]

            if groups:
                diff = max(sum(line_length for _, _, line_length in group) for group in groups) - min(sum(line_length for _, _, line_length in group) for group in groups)
                if diff < min_diff:
                    min_diff = diff
                    best_group_size = group_size

        if best_group_size is None:
            return [line_indices]

        groups = [[]]
        current_group_size = 0
        for start, end_pos, line_length in sorted(line_indices, key=lambda x: x[2], reverse=True):
            if current_group_size + line_length > best_group_size:
                if len(groups[-1]) >= min_lines_per_group:
                    groups.append([])
                    current_group_size = 0
                else:
                    continue

            groups[-1].append((start, end_pos, line_length))
            current_group_size += line_length

        if len(groups[-1]) < min_lines_per_group:
            groups = groups[:-1]

        return groups

    def shuffle_and_write_groups(self, in_file: str, out_file: str, groups: List[List[Tuple[int, int, int]]]):
        """
        Shuffles the lines within each group, and writes the shuffled groups in order to the output file.
        """
        self.random.shuffle(groups)
        for group_idx, group in enumerate(groups):
            self.random.shuffle(group)
            with open(in_file, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mem_data, open(f"{out_file}", "ab") as out:
                for idx in tqdm.tqdm(group, desc=f"Writing group {group_idx}"):
                    start, end_pos, line_length = idx
                    mem = mem_data[start:end_pos]
                    if not mem.endswith(b"\n"):
                        mem += b"\n"
                    if not mem.startswith(b"{"):
                        mem = b"{" + mem
                    out.write(mem)
                  
    def chunk_evenly(self, groups: List[List[Tuple[int, int, int]]], minimum_group_size) -> List[List[Tuple[int, int, int]]]:
        """
        Splits the groups into chunks of size equal to the smallest group, while ensuring that each chunk has at least one line.
        Larger groups are split into subchunks of the target size.
        
        Args:
            groups (List[List[Tuple[int, int, int]]]): The groups of lines to be chunked.
        
        Returns:
            List[List[Tuple[int, int, int]]]: A list of chunks, each containing a number of lines equal to the smallest group,
            except for the last chunk which may contain fewer lines if the remaining lines don't match the target size.
        """
        # Find the size of the smallest group
        smallest_group_lines = min(len(group) for group in groups)
        
        if smallest_group_lines < minimum_group_size:
            # If smallest group has only 1 line, use the specified chunk_size
            target_chunk_lines = chunk_size
        else:
            # Use the number of lines in the smallest group as the target chunk size
            target_chunk_lines = smallest_group_lines
            
        
        chunks = []
        
        for group in groups:
            group_lines = len(group)
            
            if group_lines > target_chunk_lines:
                # Split larger group into subchunks with the target number of lines
                current_subchunk = []
                subchunk_line_count = 0
                for line in group:
                    if subchunk_line_count + 1 > target_chunk_lines:
                        chunks.append(current_subchunk)
                        current_subchunk = [line]
                        subchunk_line_count = 1
                    else:
                        current_subchunk.append(line)
                        subchunk_line_count += 1
                if current_subchunk:
                    chunks.append(current_subchunk)
            else:
                # Add smaller group as a single chunk
                chunks.append(group)
        
        return chunks
