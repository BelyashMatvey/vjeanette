import argparse
import torch
from .model.model import UNet1D_Embed
from .core import InferenceRunner
from .config import DEFAULT_THRESHOLDS

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config():
    with open("pyproject.toml", "rb") as f:
        return tomllib.load(f)


def main():
    config = load_config()

    parser = argparse.ArgumentParser("Fastq Inference Tool")

    parser.add_argument('--in_file', default=None)
    parser.add_argument('--out_file', default=None)
    parser.add_argument('--model_path', default=None)

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_cores', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--logs', default='./logs/progress.log')

    args = parser.parse_args()

    # --- берем из TOML если не задано ---
    args.in_file = args.in_file or config["tool"]["vjeanette"]["test"]["input"]
    args.out_file = args.out_file or config["tool"]["vjeanette"]["test"]["output"]
    args.model_path = args.model_path or config["tool"]["vjeanette"]["test"]["model_path"]

    # --- model ---
    model = UNet1D_Embed().to(args.device)
    model.load_state_dict(torch.load(args.model_path, weights_only = True))
    model = torch.compile(model, mode="reduce-overhead")
    model = model.to(memory_format=torch.channels_last)

    # --- inference ---
    runner = InferenceRunner(model, args.device, DEFAULT_THRESHOLDS)
    runner.add_log_handler(args.logs)

    runner.run(
        fastq_file=args.in_file,
        output_csv=args.out_file,
        batch_size=args.batch_size,
        num_workers=args.n_cores
    )


if __name__ == "__main__":
    main()