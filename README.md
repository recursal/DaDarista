# DaDarista

![](DaDarista.png)

Recursal's experimental datapacker.

## Why?

DaDarista was developed to meet the needs of "We need to somehow convert jsonl files into multiple formats for different trainers!"
As such, DaDarista was born from that need.

## Dependencies

For most cases, it can be installed with the following command:

`pip install -r requirements.txt`

However, there are additional requirements if you need to use the following:

- **BinIdx** support requires `torch`.

## Usage

DaDarista takes in `.yaml` files as it's config. yaml config files are similar to RWKV-v5's datapack example found [here](https://github.com/RWKV/RWKV-infctx-trainer/blob/main/RWKV-v5/datapack-example.yaml).
Not all methods are implemented though. As such, refer to modelling.py's [DataPack](https://github.com/recursal/DaDarista/blob/main/utils/modelling.py) class for a list of supported keys and values.

## Credits

RWKV-infctx-trainer Developers: Inital Work  
Shinon: Refactor, code stripping, etc.  
m8than: Distribution by Length

## Support

While this is open sourced, we are likely not to take in any PRs or Issues as these tools are what we used internally and catered to that.
