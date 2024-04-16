import pathlib

import numpy
import orjson
import tqdm
from utils.rwkv_binidx import MMapIndexedDataset


# Taken from `make_data.py` in RWKV-v5.
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, index_file, dtype=numpy.uint16):
        self._data_file = open(out_file, "wb")
        self.index_file = index_file
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    
    def add_document(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
        self._doc_idx.append(len(self._sizes))

    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def finalize(self):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(self.index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


class BinIdxBackFlips:
    def __init__(self, file: pathlib.Path, **_) -> None:
        self.file = file

    def do(self, folder: pathlib.Path):
        print("[Binidx] Building Binidx... Have patience.")
        bin_file = folder / f"{self.file.stem}.bin"
        idx_file = folder / f"{self.file.stem}.idx"
        builder = MMapIndexedDatasetBuilder(bin_file, idx_file)
        with open(self.file,"rb") as f, tqdm.tqdm(desc="Documents Processed", it="doc",unit_scale=True) as pbar:
            chunk = f.readlines(524288000)
            while chunk:
                for line in chunk:
                    builder.add_document(numpy.array(orjson.loads(line)["input_ids"],dtype=numpy.uint16))
                    pbar.update(1)
                chunk = f.readlines(524288000)
        print("[Binidx] Finalizing...")
        builder.finalize()