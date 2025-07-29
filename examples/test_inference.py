from pathlib import Path

import pandas as pd
from llm_tools.config.config import Config
from llm_tools.dataset.dataset import Dataset
from llm_tools.llm_inference.pipeline import Pipeline
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem


def get_demo_dataset():
    return [
        {"group_id": 1, "role": "system", "content": "You are a helpful assistant."},
        {"group_id": 1, "role": "user", "content": "Hello!"},
        {"group_id": 2, "role": "system", "content": "You are a helpful assistant."},
        {"group_id": 2, "role": "user", "content": "Hello!"},
    ]


def get_paths():
    return (
        Path("../model_configs/qwen-coder.yaml"),  # Path to config
        Path("formatted_test_dataset_sub.parquet"),  # Path to dataset
    )


def main():
    cfg_pth, ds_pth = get_paths()

    # Remove assistant messages
    ds = pd.DataFrame(get_demo_dataset())
    ds = ds.loc[ds["role"] != "assistant"]

    # Check that there are no assistant messages
    print(f"Roles in ds:{ds['role'].unique()}")
    assert "assistant" not in ds["role"].to_list()

    # Create dataset and run inference
    config = Config.from_yaml(cfg_pth)

    dataset = Dataset(config)
    dataset.load_dataset(ds)

    pipe = Pipeline(config=config)
    res: list[ModelOutputItem] = pipe.run(dataset)

    # Process results
    df = pd.DataFrame([{"group": item.group_id, "content": item.text} for item in res])
    df.to_csv("AI_results.csv", index=False)


if __name__ == "__main__":
    main()
