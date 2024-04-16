import concurrent.futures as fut
from copy import deepcopy
import os
import pathlib
import typing

import datasets
import orjson
import pyarrow
import psutil
from datasets.fingerprint import generate_random_fingerprint

# NOTE: Risky!
from datasets.utils.py_utils import convert_file_size_to_int, asdict

proc = psutil.Process()
proc.cpu_affinity([i for i in range(os.cpu_count())])


def in_memory_arrow_table_from_file(filename: str) -> pyarrow.Table:
    in_memory_stream = open(filename,"rb")
    opened_stream = pyarrow.ipc.open_stream(in_memory_stream)
    # reader = pyarrow.ipc.RecordBatchFileReader(opened_stream)
    print(opened_stream.schema)
    # return pa_table


class ArrowBackFlips:
    def __init__(
        self, file: pathlib.Path, slice_size: typing.Union[int, str] = "500MB"
    ) -> None:
        """ArrowBackFlips: "Use pyarrow Directly to save files"

        This is similar to what HF's arrow datasets does but in a streaming method.

        Args:
            file (pathlib.Path): The file to be saved
            slice_size (int, str): Chunks sizes to be read before a new slice is used.
        """
        self.file = file
        self.slice_size = convert_file_size_to_int(slice_size)

    def arrow_send(
        self, arrow_table: pyarrow.Table, filename: pathlib.Path, meta: dict, schema
    ):
        # print(arrow_table.to_pandas())
        info_keys = ["features"]
        metadata = {}
        metadata["info"] = {key: meta[key] for key in info_keys}
        schema = schema
        schema = schema.with_metadata({"huggingface": orjson.dumps(metadata).decode()})
        with pyarrow.OSFile(str(filename), "wb") as sink:
            with pyarrow.ipc.RecordBatchStreamWriter(sink, schema=schema) as writer:
                print(schema)
                batch = pyarrow.record_batch(
                    [
                        pyarrow.array([arrow_table["input_ids"][0]]).cast(pyarrow.list_(pyarrow.int32())),
                        pyarrow.array([arrow_table["token_type_ids"][0]]).cast(pyarrow.list_(pyarrow.int8())),
                        pyarrow.array([arrow_table["attention_mask"][0]]).cast(pyarrow.list_(pyarrow.int8())),
                    ],
                    schema=schema,
                )
                writer.write_batch(batch)
                writer.close()
            sink.flush()
        # print("[Sent] arrow_send", str(filename))

    def arrow_batched(
        self,
        data: list[bytes],
        schema: pyarrow.Schema,
        filename: pathlib.Path,
        meta: dict,
    ):
        try:
            info_keys = ["features"]
            metadata = {}
            metadata["info"] = {key: meta[key] for key in info_keys}
            schema = schema.with_metadata({"huggingface": orjson.dumps(metadata).decode()})
            data = [orjson.loads(i) for i in data]
            with pyarrow.OSFile(str(filename), "wb") as sink:
                with pyarrow.ipc.RecordBatchStreamWriter(sink, schema=schema) as writer:
                    # arrow_table = pyarrow.Table.from_pylist(data, schema=schema)
                    # print(arrow_table)
                    batch = pyarrow.record_batch(
                        [
                            pyarrow.array([row["input_ids"] for row in data]),
                            pyarrow.array([row["token_type_ids"] for row in data]),
                            pyarrow.array([row["attention_mask"] for row in data]),
                        ],
                        schema=schema,
                        # metadata=schema,
                    )
                    # print(batch)
                    writer.write_batch(batch)
                    writer.close()
                sink.flush()
            # print("[Sent] Batched to", str(filename))
        except Exception as e:
            print(e)

    def do(self, folder: pathlib.Path):
        
        train_folder = folder / "train"
        test_folder = folder / "test"
        train_folder.mkdir(exist_ok=True, parents=True)
        test_folder.mkdir(exist_ok=True, parents=True)

        columns = datasets.Features(
            {
                "input_ids": datasets.Sequence(datasets.Value("int32",)),
                "token_type_ids": datasets.Sequence(datasets.Value("int8")),
                "attention_mask": datasets.Sequence(datasets.Value("int8")),
            }
        )
        schema = pyarrow.schema(
            [
                ("input_ids", pyarrow.list_(pyarrow.int32())),
                ("token_type_ids", pyarrow.list_(pyarrow.int8())),
                ("attention_mask", pyarrow.list_(pyarrow.int8())),
            ]
        )
        states = {
            "_data_files": [],
            # Spoof, because HF datasets apparently needs one.
            "_fingerprint": generate_random_fingerprint(nbits=32),
            "_format_columns": list(columns.keys()),
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }
        dd = folder / "dataset_dict.json"
        
        dd_info = datasets.DatasetInfo(features=columns)
        features = asdict(dd_info)
        dd.write_bytes(orjson.dumps({"splits": ["train", "test"],"features":features["features"]},))
        test = False
        chunk_idx = 0
        futures = []
        chunk_lines_test = []
        with open(self.file, "rb") as f, fut.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            test_set = orjson.loads(f.readline())
            f.seek(0)
            saved_files = []
            chunk_lines_test = f.readlines(self.slice_size)
            while chunk_lines_test:
                # Firstwrite test train folder
                if not test:
                    # pytorch lightning trainer is picky to have 1 value for train.
                    # print(len(data["input_ids"]), len(data["token_type_ids"]), len(data["attention_mask"]))
                    pytable = pyarrow.Table.from_pylist([test_set])
                    test = True
                    test_states = deepcopy(states)
                    test_states["_data_files"].append(
                        {"filename": f"data-{str(0).zfill(5)}.arrow"}
                    )
                    (test_folder / "state.json").write_bytes(
                        orjson.dumps(test_states, option=orjson.OPT_INDENT_2)
                    )
                    filename = test_folder / f"data-{str(0).zfill(5)}.arrow"
                    self.arrow_send(pytable, filename, features, schema)
                    dataset_json = test_folder / "dataset_info.json"
                    dataset_json.write_bytes(
                        orjson.dumps(features, option=orjson.OPT_INDENT_2)
                    )
                    # print(pytable)
                # Iter over chunks and write to pool
                filename = train_folder / f"data-{str(chunk_idx).zfill(5)}.arrow"
                print("submit to pool")
                saved_files.append(filename.name)
                future = executor.submit(
                    self.arrow_batched, chunk_lines_test, schema, filename, features
                )
                futures.append(future)
                print("nxt chunk")
                chunk_idx += 1
                cpu_count = 6 if not os.cpu_count() else os.cpu_count()
                # Pause if our futures are more than cpu counts.
                if len(futures) >= cpu_count:
                    for future in futures:
                        future.result()
                    futures = []
                chunk_lines_test = f.readlines(self.slice_size)
            # Pause and wait for all futures to be done.
            for future in futures:
                future.result()
            futures = []

            train_states = deepcopy(states)
            # Write states
            for fn in saved_files:
                train_states["_data_files"].append({"filename": fn})
            (train_folder / "state.json").write_bytes(
                orjson.dumps(train_states, option=orjson.OPT_INDENT_2)
            )
            # Write features
            dataset_json = train_folder / "dataset_info.json"
            dataset_json.write_bytes(orjson.dumps(features, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def arrow(file: pathlib.Path):
        ArrowBackFlips(file).do(file.resolve().parent)

    @app.command()
    def schema(file: pathlib.Path):
        in_memory_arrow_table_from_file(str(file))

    app()
