import pathlib
import typing
import pydantic
import enum


class MixinMode(enum.Enum):
    shuffle = "shuffle"
    dist_by_length = "dist_by_length"  # Smart mix mode that mixes the data into groups of similar sizes and then evenly distributes these groups.
    concat = "concat"
    batch = "batch"  # NOTE: Batch isn't supported.


class DataPack(pydantic.BaseModel):
    data_path: pathlib.Path
    batchsize: int
    mixing_mode: MixinMode
    mixed_batch_percentage: typing.Optional[float] = 0.0

    auto_mask_user_worldml: bool = False

    dist_by_length_min_lines_per_group: int = 4096
    dist_by_length_min_subchunk_size: int = 1024  # this is after subchunking, at least 4 groups will be created for each length.

    packing_batchsize: int = 64
    dataset_weight: float = 1.0
    inverse_shuffle: typing.Optional[bool] = False

    # Configues to control how DataCompiler does multiproessing.

    # Determine the number of cores available to adjust batch sizes accordingly. Exercise caution when modifying.
    batch_scale: typing.Optional[int] = None

    # Number of simultaneous processes to execute across all subchunks.
    process_subchunks: typing.Optional[int] = None

    # Number of tokenizing threads to execute per subchunk.
    # This only affects the batch size for `.iter()` during `thread_tokenizing`.
    thread_tokenizing: typing.Optional[int] = None

    # Number of processes to execute simultaneously during rechunking.
    process_rechunking: typing.Optional[int] = None


class DatapathStorageOpts(pydantic.BaseModel):
    key: str
    secret: str
    endpoint_url: str


class DatasetConfig(pydantic.BaseModel, extra="forbid"):
    # The root dataset?
    data_path: typing.Annotated[pathlib.Path, pydantic.DirectoryPath]

    # === Test Splits ===
    # test_split:
    # (int): no. of samples
    # (float): As a % of the total number of data samples
    test_split: typing.Union[int, float] = pydantic.Field(None, ge=0)
    test_split_shuffle: bool

    # Sets the dataset tokenizer.
    # Valid options can be: `world`, `neox` (neox isn't supported lol)
    # Custom tokenizers are supported by provided a HF Tokenizer name/path.
    tokenizer: str = "world"

    # Loader to use.
    # accelerated (Stricter checks), legacy (HF based)
    loader: str = "legacy"

    # === Token rechunking ===
    # the name is a misnomer.

    # The number of tokens to chunk over.
    text_rechunk_size: int = 2048

    # Force rechunking over all keys
    # in data_new.py, it enable rechunking or not.
    text_rechunk_force: bool = True

    # disable the automated text rechunking for text files
    text_rechunk_auto: bool = True

    # === Optional Sub Dataset configuration ===
    name: str = ""
    source: typing.Annotated[pathlib.Path, pydantic.DirectoryPath] = pathlib.Path("")
    source_data_dir: str = ""

    # Offsets / Seeks
    offset: typing.Union[float, int] = 0

    # Key to use to tokenize. Defaults to `text`.
    key: str = "text"

    # Deprecated. Included to prevent conflicts
    # seems like in pydantic 2.6, there is deprecated arguments.
    packing_enable: bool = False
    reverse_train_dataset_before_save: bool = False

    # TODO: Find out what this is. - Shinon
    dataset_split: str = "train"
