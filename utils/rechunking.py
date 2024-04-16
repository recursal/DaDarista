import multiprocessing
import pathlib
import platform
import subprocess
import traceback

import natsort
import orjson
import tqdm
from functional import line_counter


def rechunk_single(input_file: pathlib.Path, output_file: pathlib.Path, rechunk_size):
    """Write the rechunked thingy
    TODO: Write some actual docs once I'm not so high.

    Args:
        rechunked (_type_): _description_
        list_data (_type_): _description_
        rechunk_size (_type_): _description_
        parser_queue (_type_): _description_
    """
    with open(output_file, "wb") as rechunk_file, open(input_file, "rb") as input_fp:
        lines = input_fp.readlines(1048576 * 50)
        tokens = []
        while lines:
            for line in lines:
                tokens.extend(orjson.loads(line)["input_ids"])
                while len(tokens) > rechunk_size:
                    sliced = tokens[:rechunk_size]
                    linebuffer = orjson.dumps(
                        {
                            "input_ids": sliced,
                            "token_type_ids": [0] * rechunk_size,
                            "attention_mask": [1] * rechunk_size,
                        }
                    )
                    linebuffer += b"\n"
                    rechunk_file.write(linebuffer)
                    linebuffer = b""
                    tokens = tokens[rechunk_size:]
            lines = input_fp.readlines(1048576 * 50)
        while len(tokens) > rechunk_size:
            sliced = tokens[:rechunk_size]
            linebuffer = orjson.dumps(
                {
                    "input_ids": sliced,
                    "token_type_ids": [0] * rechunk_size,
                    "attention_mask": [1] * rechunk_size,
                }
            )
            linebuffer += b"\n"
            rechunk_file.write(linebuffer)
            linebuffer = b""
            tokens = tokens[rechunk_size:]
        sliced = tokens[:rechunk_size]
        sliced_size = len(sliced)
        linebuffer = orjson.dumps(
            {
                "input_ids": sliced + [0] * (rechunk_size - sliced_size),
                "token_type_ids": [0] * len(sliced),
                "attention_mask": [1] * len(sliced)
                + [0] * (rechunk_size - sliced_size),
            }
        )
        linebuffer += b"\n"
        rechunk_file.write(linebuffer)
        linebuffer = b""


def err_cb(err: BaseException):
    traceback.print_exception(err)


def rechunk_final(
    file: pathlib.Path,
    rechunked: pathlib.Path,
    rechunk_size: int,
    process_rechunking: int,
):
    """Rechunks (by tokens) the selected file with...
    TODO: Write some actual docs once I'm not so high.

    Args:
        file (pathlib.Path): _description_
        rechunked (pathlib.Path): _description_
        rechunk_size (int): _description_
    """

    file = file.resolve()
    lines_per_chunk = int(line_counter(file) // (process_rechunking))
    print("LPC", (process_rechunking), lines_per_chunk)
    if "linux" in platform.system().lower():
        subprocess.call(
            [
                "split",
                "-l",
                str(lines_per_chunk),
                "--additional-suffix",
                ".jsonl",
                str(file),
                f"{file.parent}/rechunk-split-",
            ]
        )
        with multiprocessing.Pool(processes=process_rechunking) as pool:
            so = natsort.os_sorted(list(file.parent.glob("rechunk-split*.jsonl")))
            corr_names = [
                f_i.with_name(f"rechunk-done-split-{f_i.stem.split('-')[-1]}.jsonl")
                for f_i in so
            ]
            print(so, corr_names)
            print("Creating Magic")
            fns = [
                pool.apply_async(
                    rechunk_single,
                    args=(z_single, corr_names[idx], rechunk_size),
                    error_callback=err_cb,
                )
                for idx, z_single in enumerate(so)
            ]
            print("Waiting for magic to finish...")
            for fn in tqdm.tqdm(fns):
                fn.wait()
            for i in so:
                i.unlink()
            print("Combine Magic")
            with open(rechunked, "wb") as f_out:
                for file in corr_names:
                    with open(file, "rb") as f:
                        for line in tqdm.tqdm(f, desc=f"{file.name}"):
                            f_out.write(line.rstrip() + b"\n")
            for i in corr_names:
                i.unlink()
