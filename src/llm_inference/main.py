import argparse
from pathlib import Path
from llm_inference.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LLM Inference")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to config file."
    )
    parser.add_argument(
        "-d,--dataset", type=Path, required=True, help="Path to csv dataset."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = Pipeline(config_path=args.config)
    pipeline.run(args.dataset)
    pipeline.save_results(Path("./results.csv"))


if __name__ == "__main__":
    main()
