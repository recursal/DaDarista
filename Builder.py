import enum
import pathlib
import typer

import utils.modelling as pydantic_models

from Compiler import DataCompiler

app = typer.Typer()


class BackFlipMethod(enum.Enum):
    Arrow = "Arrow"
    Binidx = "binidx"


@app.command()
def prepare(
    config: pathlib.Path,
    with_backflips: bool = False,
    backflip: BackFlipMethod = BackFlipMethod.Arrow.value,
):
    """Packs the data into the datapack

    Args:
        config (pathlib.Path): The config datapack config file
        with-backflips (bool, optional): Converts the final jsonl to one of the supported backflipping formats. Defaults to False.
        backflip (BackFlipMethod, optional): Selects which backflip method to use. Defaults to "Arrow", Can be [`binidx`, `binidx_zero`]

    Notes:
        binidx: When the backflip output is binidx, ensure that rechunking is disabled.
          This appears to be the behavior in `make_data.py` from the original RWKV-LM repo.

    Raises:
        ValueError: Attempted to set a non existant/unconfigured/unseen attribute within the dataset config.

    Returns:
        None: Returns nothing once done.
    """
    if config.is_file() and any(
        [config.suffix.lower().endswith("yaml"), config.suffix.lower().endswith("yml")]
    ):
        import yaml

        with open(config, "r") as f:
            datapack_config = yaml.safe_load(f)
        prechecks = [
            "datapack" not in datapack_config,
            "default" not in datapack_config,
            "dataset" not in datapack_config,
        ]
        if any(prechecks):
            typer.secho(
                f"Datapack precheck failed: {prechecks} [datapack, default, dataset] is missing.",
                fg="red",
            )
            return -1
        datapack = pydantic_models.DataPack.model_validate(datapack_config["datapack"])
        default_config = pydantic_models.DatasetConfig.model_validate(
            datapack_config["default"]
        )
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

        dc = DataCompiler(None)
        # Does the actual datapacking.

        if backflip == BackFlipMethod.Binidx and default_config.text_rechunk_force:
            print("########################################")
            print("# Attempting to rechunk when data output format is binidx.")
            print("# (`text_rechunk_force` == True && `backflip` == Binidx)")
            print("# Disable rechunking to use binidx.")
            print("########################################")
            return

        dataset, final_file = dc.prepare_datapack(
            datapack, default_config, subsets, skip_unprocessed=False
        )
        if with_backflips and backflip == BackFlipMethod.Arrow:
            print("[Arrow Backflips]")
            from Backflippers.ArrowBackflip import ArrowBackFlips

            ArrowBackFlips(final_file).do(final_file.resolve().parent)
            final_file.rename(default_config.data_path / final_file.name)
        elif with_backflips and backflip == BackFlipMethod.Binidx:
            print("[BinIdx Backflips]")
            from Backflippers.BinIdxBackflip import BinIdxBackFlips

            BinIdxBackFlips(final_file).do(final_file.resolve().parent)
            final_file.rename(default_config.data_path / final_file.name)


if __name__ == "__main__":
    app()
