"""
Trie Tokenizer used for RWKV.

The original implementation can be found here:

https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/rwkv_tokenizer.py
https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer
"""

try:
    from world_tokenizer_cpp import TRIE_TOKENIZER as CppTrieTokenizer

    cpp_tokenizer = True
except ImportError:
    pass
import ast
import gzip
import lzma
import pathlib
from typing import List, TextIO

import tqdm
from .GenericTokenizer import GenericTokenizer


class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to: list
    values: set

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr != None:
            if fr.ch != None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class TrieTokenizerException(Exception):
    pass


class WorldTrieTokenizer(GenericTokenizer):
    def __init__(
        self,
        tokenizer_file: pathlib.Path,
        try_cpp=False,
        vocab_size: int = 65536,
    ) -> None:
        if not tokenizer_file.is_file():
            raise Exception(
                "WorldTokenizer expects a vocab file. But was not provided."
            )
        self.tokenizer_file = tokenizer_file
        self.vocab_size = vocab_size
        self.cpp_tokenizer = None

        self.idx2token = {}
        self.token2idx = {}

        if try_cpp and cpp_tokenizer:
            print("[worldTokenizer] Trying CPP...")
            if self.prepare_cpp():
                print("CPP WorldTokenizer loaded successfully.")
        self.prepare_trie()

    @property
    def get_vocab_size(self):
        return self.vocab_size

    @property
    def get_vocab(self):
        return self.idx2token

    def prepare_cpp(self):
        self.cpp_tokenizer = CppTrieTokenizer(str(self.tokenizer_file))
        return self.cpp_tokenizer

    def prepare_trie(self):
        if self.tokenizer_file.suffix.endswith(".gz"):
            fp: TextIO = gzip.open(self.tokenizer_file, "r", encoding="utf-8")
        elif self.tokenizer_file.suffix.endswith(".xz"):
            fp: TextIO = lzma.open(self.tokenizer_file, "r", encoding="utf-8")
        else:
            fp: TextIO = open(self.tokenizer_file, "r", encoding="utf-8")
        with fp as file:
            for line in tqdm.tqdm(file, desc="Prepare Tokenizer..."):
                segments = line.split(" ")
                idx = int(segments[0])
                token_size = int(segments[-1])
                token = " ".join(segments[1:-1]).strip(" ")
                token = ast.literal_eval(token)
                if isinstance(token, str):
                    token = token.encode("utf-8")
                if len(token) != token_size:
                    raise TrieTokenizerException(
                        f"idx {idx}, {len(token)} does not match expected token count of:{token_size}"
                    )
                self.idx2token[idx] = token
        # To match behavior. In anycase, Cpp Tokenizer always adds an eot tokens.
        self.idx2token[0] = b"<|endoftext|>"
        self.trie_cls = TRIE()
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)
        for t, i in self.token2idx.items():
            _ = self.trie_cls.add(t, val=(t, i))
        # Free some memory.
        del self.token2idx

    def encode(self, contents: str | bytes):
        if isinstance(contents, str):
            contents = contents.encode("utf-8")
        idx: int = 0
        tokens = []
        while idx < len(contents):
            _idx: int = idx
            idx, _, values = self.trie_cls.find_longest(contents, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decode(self, tokens: List[bytes], as_str: bool = False) -> str | bytes:
        if as_str:
            return b"".join(map(lambda i: self.idx2token[i], tokens)).decode("utf-8")
        else:
            return b"".join(map(lambda i: self.idx2token[i], tokens))

    @property
    def encode_fn(self):
        return self.cpp_tokenizer if self.cpp_tokenizer else self

    def tokenize(self, contents: List[str] | str):
        """Tokenizes a string of contents.

        Args:
            contents (typing.Union[typing.List[str], str]): Tokenizes either a list of UTF-8 strings or a single UTF-8 string.

        Raises:
            NotImplementedError: The method is not implemented for the current tokenizer.

        Notes:
            When implementing, take note that this functions will be called more than once across multiple processes / threads. As such, ensure that this function is threadsafe.
        """
        if isinstance(contents, str):
            # Tokenize the entire string
            encoded = self.encode_fn.encode(contents)
            return {
                "input_ids": encoded,
                "token_type_ids": [0] * len(encoded),
                "attention_mask": [1] * len(encoded),
            }
        elif isinstance(contents, list):
            # Tokenize list contents.
            id_arr = []
            type_arr = []
            mask_arr = []
            for str_contents in contents:
                enc_str = self.encode_fn.encode(str_contents)
                id_arr.append(enc_str)
                type_arr.append([0] * len(enc_str))
                mask_arr.append([1] * len(enc_str))
            return {
                "input_ids": id_arr,
                "token_type_ids": type_arr,
                "attention_mask": mask_arr,
            }
        else:
            raise ValueError(
                f"Invalid type to tokenize. Expected List[str] / str. Got: {type(contents)}"
            )
