import concurrent.futures as conc
import logging
import os
import pathlib
import typing

import orjson
import tqdm
from datasets import (
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from utils.modelling import DataPack, DatasetConfig, MixinMode
from tokenizers.DatasetTokenizer import DatasetTokenizer
from tokenizers.TrieTokenizer import WorldTrieTokenizer
from utils.rechunking import rechunk_final
from utils.shufflers import dist_by_length_mmapped, shuffle_mmaped


class ExperimentalLoader:
    def __init__(self, file: pathlib.Path, parse_json: bool = False) -> None:
        if not file.is_file():
            raise ValueError(f"Expecting: {file}, but it does not exist.")
        self.file = file
        self.json = parse_json

    def __iter__(self):
        with open(self.file, "rb") as f:
            for line in f:
                if self.json:
                    yield orjson.loads(line)
                else:
                    yield line


class DataCompiler:
    def __init__(
        self, tokenizer: typing.Optional[pathlib.Path], log_level=logging.INFO
    ) -> None:
        """DataCompiler compiles... well data.

        This is essentially a slimmed and trimmed down version of the one present in data.py

        Args:
            tokenizer (typing.Optional[pathlib.Path]): The tokenizer file to use. If `None`, uses the default tokenizer bundled.
            log_level (_type_, optional): The logging level to log stuff to. Defaults to logging.INFO. Recommended to leave it as is to understand where it is
        """
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("DataCompiler")
        self.logger.info("[PRP] WorldTokenizer")
        self.tokenizer_file = tokenizer
        self.logger.info("[OK] WorldTokenizer")

        # Get the number of cores to see how much batch sizes we can use per run.
        # This additionally affects the batching count to be used. modify with caution.
        # Scale the number of threads/processes to use.
        # This now only affect the batch size to .iter() for for `thread_tokenizing`.
        self.batch_scale = 2
        # How many processes to run simultaneously across all subchunks?
        self.process_subchunks = os.cpu_count() // 10
        # How many tokenizing threads to run per each subchunk?
        self.thread_tokenizing = 2
        # How many processes to run simultaneously when rechunking?
        # Note: This also affects individual split sizes during rechunk
        self.process_rechunking = 30

    def prepare_datapack_from_config(
        self, config: typing.Union[pathlib.Path, str]
    ) -> IterableDatasetDict:
        """Prepares datapack from the config file

        Args:
            config (typing.Union[pathlib.Path,str]): The config file. If it's a string, attempt to parse it as a file

        Raises:
            ValueError: If the config file cannot be found.
            ValueError: If there is any additional keys in the datapack/default/dataset keys that are not allowed to be set.

        Returns:
            IterableDatasetDict: If the data goes through.
        """
        import yaml

        if isinstance(config, str):
            if not pathlib.Path(config).is_file():
                raise ValueError(f"{config} is not a valid file!")
            else:
                config = pathlib.Path(config)

        with open(config, "r") as f:
            datapack_config = yaml.safe_load(f)
        prechecks = [
            "datapack" not in datapack_config,
            "default" not in datapack_config,
            "dataset" not in datapack_config,
        ]
        if any(prechecks):
            print(
                f"Datapack precheck failed: {prechecks} [datapack, default, dataset] is missing."
            )
            return None
        datapack = DataPack.model_validate(datapack_config["datapack"])
        default_config = DatasetConfig.model_validate(datapack_config["default"])
        subsets = []
        for subset in datapack_config["dataset"]:
            subset: dict = subset
            base_subset = default_config.model_copy()
            for k, v in subset.items():
                if not hasattr(base_subset, k):
                    raise ValueError(f"Attempted to set an unknown attribute: {k}")
                setattr(base_subset, k, v)
                # subset[] = v
            subsets.append(base_subset)
        print(f"Found: {len(subsets)}.")
        iterable_prepared = self.prepare_datapack(datapack, default_config, subsets)
        if iterable_prepared:
            return IterableDatasetDict(
                {"train": iterable_prepared, "test": iterable_prepared.take(1)}
            )

    def line_counter(self, file: pathlib.Path):
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

    def dataset_counter(self, dataset: IterableDataset, target: int) -> int:
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

    def prepare_dataset(
        self,
        dataset: DatasetConfig,
        default: DatasetConfig,
        force_test: bool = False,
        skip_unprocessed: bool = False,
    ) -> typing.Optional[pathlib.Path]:
        """Prepares each dataset with the config specified.

        Args:
            dataset (DatasetConfig): The dataset config. Should not be used directly.
            force_test (bool, optional): As iterable map functions are lazily applied, this calls .iter(). do not use outside of testing! Defaults to False.
            skip_unprocessed (bool, optional): Skips tokenizing already existing files. Do not use outside of testing!

        Raises:
            ValueError: Missing `source` value
            NotImplementedError: offsets have not been implemented. Woops.
            ValueError: Raises an error if `dataset_split` value does not exist.

        Returns:
            pathlib.Path: The tokenized dataset file.
        """

        # Rewrite arguments from model to load_dataset compatible ones.
        # We really should just do overrides instead.
        # - Shinon

        dataset_args = {
            "path": dataset.source,
            "streaming": True,
            "data_dir": dataset.source_data_dir
            if dataset.source_data_dir
            else dataset.data_path,
        }
        if dataset.source is None:
            raise ValueError(
                "`source` field is missing. Did you forget to set it in default?"
            )
        streaming_dataset_dict: IterableDatasetDict = load_dataset(**dataset_args)
        # print(streaming_dataset)
        if dataset.offset not in [0, 0.0]:
            if isinstance(dataset.offset, float):
                print(streaming_dataset_dict)
                raise NotImplementedError()
            elif isinstance(dataset.offset, int):
                print(streaming_dataset_dict)
                raise NotImplementedError()
        if dataset.dataset_split not in streaming_dataset_dict:
            print(list(streaming_dataset_dict.keys()), "Available as valid splits.")
            raise ValueError(
                f"`{dataset.dataset_split}` split does not exist in dataset!"
            )
        streaming_dataset: IterableDataset = streaming_dataset_dict[
            dataset.dataset_split
        ]

        if not isinstance(streaming_dataset, IterableDataset):
            raise ValueError(
                f"`{type(streaming_dataset)}` for `streaming_dataset` was unexpected."
            )

        streaming_dataset = streaming_dataset.filter(lambda example: example["text"])

        # We check here if we have an existing file since tokenizing is slow.
        if (pathlib.Path(default.data_path) / f"{dataset.name}.jsonl").exists():
            self.logger.info("Found tokenized file. Checking if line count matches...")
            lines = self.line_counter(
                pathlib.Path(default.data_path) / f"{dataset.name}.jsonl"
            )
            streamed_lines = self.dataset_counter(streaming_dataset, lines)
            if lines == -1:
                self.logger.info("Re-tokenizing. Tokenized is too small!")
                # Exited early because the lines in the tokenized version is too small.
                pass
            elif lines == streamed_lines:
                self.logger.info(
                    "Tokenized file matches pre-tokenized file line count. Skipping..."
                )
                return pathlib.Path(default.data_path) / f"{dataset.name}.jsonl"
            else:
                self.logger.info(
                    f"Re-tokenizing. {lines} does not match {streamed_lines}"
                )
        # Debug checks.
        if skip_unprocessed:
            return None
        # Prepare the concurrent pool here as well as the tokenizer.
        # These aren't serializable across processes.
        global_tokenizer = DatasetTokenizer()

        if dataset.tokenizer.lower() == "world":
            if self.tokenizer_file is None:
                self.logger.info("World vocab missing. Using default.")
                self.tokenizer_file = (
                    pathlib.Path(__file__).resolve().parent
                    / "vocab"
                    / "RWKVVocab.txt.gz"
                )
            global_tokenizer.set_tokenizer(
                WorldTrieTokenizer(self.tokenizer_file),
                self.thread_tokenizing,
                self.batch_scale,
            )
        else:
            raise Exception("Tokenizer not supported")
        tok_unshuff = (
            pathlib.Path(default.data_path) / f"{dataset.name}_tokenized_preshuff.jsonl"
        )
        self.logger.info("Prepare Tokenized...")

        global_tokenizer.tokenize_dataset(streaming_dataset, tok_unshuff, dataset.key)

        # do shuffling.
        # We don't check for the shuffled file since we are sure that
        # we will need to rebuild it.
        tok_shuffled = pathlib.Path(default.data_path) / f"{dataset.name}.jsonl"
        shuffle_mmaped(tok_unshuff, tok_shuffled, 101, self.logger)
        dataset_args = {
            **dataset_args,
            "data_dir": default.data_path,
            "data_files": [
                str(pathlib.Path(default.data_path) / f"{dataset.name}.jsonl")
            ],
        }
        # remove unshuffled file.
        if tok_unshuff.is_file():
            tok_unshuff.unlink()
        self.logger.info(f"[OK] Tokenized {dataset.source_data_dir}.")
        return pathlib.Path(default.data_path) / f"{dataset.name}.jsonl"

    def prepare_datapack(
        self,
        datapack: DataPack,
        default: DatasetConfig,
        datasets: typing.List[DatasetConfig],
        skip_unprocessed: bool = False,
    ) -> typing.Optional[typing.Union[IterableDatasetDict, pathlib.Path]]:
        """_summary_

        Args:
            datapack (DataPack): The datapack config
            default (DatasetConfig): The default dataset config
            datasets (typing.List[DatasetConfig]): List of dataset configs to be used
            skip_unprocessed (bool, optional): Skips unprocessed datasets. Only used for debugging. Defaults to False.

        Raises:
            NotImplementedError: Specified mixing mode is not compatible

        Returns:
            typing.Optional[IterableDataset]: _description_
        """
        data_path = pathlib.Path(datapack.data_path)
        if (data_path / "final.jsonl").is_file():
            pass
        if not data_path.is_dir():
            self.logger.info(f"Making path at: {datapack.data_path}")
            data_path.mkdir(exist_ok=True, parents=True)
        if not pathlib.Path(default.data_path).is_dir():
            self.logger.info(f"Making path at: {datapack.data_path}")
            pathlib.Path(default.data_path).mkdir(exist_ok=True, parents=True)
        iter_prepared = []

        if datapack.batch_scale:
            self.logger.info(f"Changing batch_scale to: {datapack.batch_scale}")
            self.batch_scale = datapack.batch_scale
        if datapack.process_subchunks:
            self.logger.info(
                f"Changing process_subchunks to: {datapack.process_subchunks}"
            )
            self.batch_scale = datapack.process_subchunks
        if datapack.thread_tokenizing:
            self.logger.info(
                f"Changing thread_tokenizing to: {datapack.thread_tokenizing}"
            )
            self.batch_scale = datapack.thread_tokenizing
        if datapack.process_rechunking:
            self.logger.info(
                f"Changing process_rechunking to: {datapack.process_rechunking}"
            )
            self.batch_scale = datapack.process_rechunking

        with conc.ProcessPoolExecutor(max_workers=self.process_subchunks) as pool:
            futures = []
            for dataset in datasets:
                self.logger.info(f"[PRP] Found {dataset.source_data_dir}")
                pool_future = pool.submit(
                    self.prepare_dataset,
                    dataset,
                    default,
                    skip_unprocessed=skip_unprocessed,
                    force_test=False,
                )
                futures.append(pool_future)
            for future in futures:
                print(future)
                try:
                    r = future.result()
                except Exception as e:
                    print(e)
                print(r)
                iter_prepared.append(r)
            # tokenized =
        # Concate all prepared datasets
        unshuffled = data_path / "final_unshuff.jsonl"
        with open(unshuffled, "wb") as f, tqdm.tqdm(
            desc="Final.jsonl Compile", unit="line", unit_scale=True
        ) as pbar:
            # When concating files, we just skip over running through HF a second time.
            for final_file in iter_prepared:
                with open(final_file, "rb") as dset_file:
                    for line in dset_file:
                        f.write(line.rstrip() + b"\n")
                        pbar.update(1)

        # If Inverse shuffle + Mixing Mode == Shuffle and text_rechunk_force is True...
        self.shuffle(datapack, default, data_path, unshuffled)

    def shuffle(
        self,
        datapack: DataPack,
        default: DatasetConfig,
        data_path: pathlib.Path,
        unshuffled: pathlib.Path,
    ) -> typing.Optional[typing.Union[IterableDatasetDict, pathlib.Path]]:
        """Shuffles the final dataset

        Args:
            datapack (DataPack): The datapack config
            default (DatasetConfig): The default dataset config
            data_path (pathlib.Path): The datapack content.
            unshuffled (pathlib.Path): The unshuffled pathlib.

        Raises:
            NotImplementedError: The shuffle

        Returns:
            typing.Optional[typing.Union[IterableDatasetDict, pathlib.Path]]: _description_
        """
        if (
            datapack.inverse_shuffle
            and datapack.mixing_mode == MixinMode.shuffle
            and default.text_rechunk_force
        ):
            self.logger.info("Rechunking before shuffling...")
            self.logger.info(
                f"Enabled text_rechunk_force with {default.text_rechunk_size} as size. Rechunking the entire file..."
            )
            rechunked_pre = data_path / "final_unrechunked.jsonl"
            if rechunked_pre.is_file():
                rechunked_pre.unlink()
            rechunk_pre = unshuffled.rename(rechunked_pre)
            rechunk_final(
                rechunk_pre,
                unshuffled,
                default.text_rechunk_size,
                self.process_rechunking,
            )
            rechunk_pre.unlink()
            self.logger.info("Shuffling contents...")
            # Save the entire thing to jsonl and then shuffle based on that.
            # concated.iter
            shuffled = data_path / "final.jsonl"
            shuffle_mmaped(unshuffled, shuffled, 101, self.logger)
            unshuffled.unlink()
            return load_dataset(
                "json", data_files=str(shuffled), streaming=True
            ), shuffled

        # If it's not inverse shuffle but generic shuffle...
        if datapack.mixing_mode == MixinMode.shuffle:
            self.logger.info("Shuffling contents...")
            # Save the entire thing to jsonl and then shuffle based on that.
            # concated.iter
            shuffled = data_path / "final.jsonl"
            shuffle_mmaped(unshuffled, shuffled, 101, self.logger)
            unshuffled.unlink()
            if default.text_rechunk_force:
                self.logger.info(
                    f"Enabled text_rechunk_force with {default.text_rechunk_size} as size. Rechunking the entire file..."
                )
                rechunked_pre = data_path / "final_unrechunked.jsonl"
                if rechunked_pre.is_file():
                    rechunked_pre.unlink()
                rechunk_pre = shuffled.rename(rechunked_pre)
                rechunk_final(
                    rechunk_pre,
                    shuffled,
                    default.text_rechunk_size,
                    self.process_rechunking,
                )
                rechunk_pre.unlink()
            return load_dataset(
                "json", data_files=str(shuffled), streaming=True
            ), shuffled

        # If it's concat...
        elif datapack.mixing_mode == MixinMode.concat:
            # Return as is.
            self.logger.info("Returning unshuffled contents...")
            shuffled = unshuffled.rename(data_path / "final.jsonl")
            if default.text_rechunk_force:
                self.logger.info(
                    f"Enabled text_rechunk_force with {default.text_rechunk_size} as size. Rechunking the entire file..."
                )
                rechunked_pre = data_path / "final_unrechunked.jsonl"
                if rechunked_pre.is_file():
                    rechunked_pre.unlink()
                rechunk_pre = shuffled.rename(rechunked_pre)
                rechunk_final(
                    rechunk_pre,
                    shuffled,
                    default.text_rechunk_size,
                    self.process_rechunking,
                )
                rechunk_pre.unlink()
            return load_dataset(
                "json", data_files=str(shuffled), streaming=True
            ), shuffled
        # If mixing mode is requested to use length distribution (nathan's)
        elif datapack.mixing_mode == MixinMode.dist_by_length:
            # Return as is.
            self.logger.info("Smartly chunking by length and distributing evenly...")
            if default.text_rechunk_force:
                self.logger.info("Rechunking not supported for length_dist.")

            smartchunk = data_path / "final.jsonl"

            dist_by_length_mmapped(
                unshuffled,
                smartchunk,
                102,
                datapack.dist_by_length_min_lines_per_group,
                datapack.dist_by_length_min_subchunk_size,
                self.logger,
            )
            unshuffled.unlink()
            return load_dataset(
                "json", data_files=str(smartchunk), streaming=True
            ), smartchunk
        else:
            raise NotImplementedError()
