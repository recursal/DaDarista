import logging
import pathlib
from datasets import IterableDataset
import orjson
import tqdm
import concurrent.futures as conc

from .GenericTokenizer import GenericTokenizer


class DatasetTokenizer:
    def __init__(self, log_level=logging.INFO) -> None:
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("DatasetTokenizer")

    def tokenize_field(self, x: dict, key: str):
        if key not in x:
            self.logger.warning(
                f"Expected {key} to exist, but it didn't for this data: {x}. Skipping!"
            )
            return x
        if isinstance(x[key], list):
            response = self.concurrent.map(self.tokenizer.tokenize, x[key])
            tokenized = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
            for r in response:
                for k, v in r.items():
                    tokenized[k].append(v)
            return response
        return self.tokenizer.tokenize(x[key])

    def set_tokenizer(
        self, tokenizer: GenericTokenizer, thread_tokenizing: int, batch_scale: int
    ):
        """Sets the tokenizer for the dataset tokenizer.
        
        Tokenizer must be an instance of a subclass'd GenericTokenizer!

        Args:
            tokenizer (GenericTokenizer): The tokenizer to use.
            thread_tokenizing (int): The number of concurrent threads to create.
            batch_scale (int): Scales the batch (batch * thread_tokenizing)
        """
        self.tokenizer = tokenizer
        self.thread_tokenizing = thread_tokenizing
        self.batch_scale = batch_scale
        self.concurrent = conc.ThreadPoolExecutor(thread_tokenizing)

    def tokenize_dataset(
        self, streaming_dataset: IterableDataset, export_file: pathlib.Path, text_key:str
    ):
        tokenized = streaming_dataset.map(
            self.tokenize_field,
            fn_kwargs={"key": text_key},
        )
        tokenized = tokenized.select_columns(
            ["input_ids", "token_type_ids", "attention_mask"]
        )
        
        self.logger.info("Prepare Tokenized...")
        if not export_file.is_file():
            with open(export_file, "wb") as f, tqdm.tqdm(
                desc="Tokenizing", unit_scale=True, dynamic_ncols=True
            ) as pbar, tqdm.tqdm(
                desc="Token Stream", unit_scale=True, dynamic_ncols=True
            ) as pbar_tok:
                for batch in tokenized.iter(
                    batch_size=self.thread_tokenizing * self.batch_scale
                ):
                    rewrite = [
                        {
                            "input_ids": batch["input_ids"][i],
                            "token_type_ids": batch["token_type_ids"][i],
                            "attention_mask": batch["attention_mask"][i],
                        }
                        for i in range(len(batch["input_ids"]))
                    ]
                    pbar_tok.update(sum([len(i["input_ids"]) for i in rewrite]))
                    for row in self.concurrent.map(orjson.dumps, rewrite):
                        pbar.update(1)
                        f.write(row + b"\n")
