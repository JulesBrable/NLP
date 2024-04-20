import yaml
import json
import pandas as pd
import argparse
import re
import numpy as np


def get_params(
    filepath: str = 'conf/params.yaml'
        ):
    """
    Import parameters grid
    """
    with open(filepath, 'rb') as f:
        conf = yaml.safe_load(f.read())
    return conf


def get_data(filepath: str):
    suffix = filepath.split('.')[-1].lower()
    if suffix in ["yaml", "yml"]:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    elif suffix == "json":
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif suffix == "csv":
        data = pd.read_csv(filepath)
    else:
        print("Invalid data type")
        return
    return data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--es",
        type=bool,
        default=False,
        help="Whether to finetune models with early stopping"
    )
    args = parser.parse_args()
    return args


def update_json(json_file, key, value):
    j = get_data(json_file)
    j[key] = value
    with open(json_file, 'w') as f:
        json.dump(j, f, indent=4)


def filter_one_age(age):
    if isinstance(age, str):
        if re.match(r"^[0-9]+$", age):
            return int(age)
        else:
            return np.nan
    else:
        try:
            return int(age)
        except ValueError:
            return np.nan


def display_results(results):
    for label, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"Class {label}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name.capitalize()}: {metric_value:.2f}")
        else:
            print(f"{label.capitalize()}: {metrics:.2f}")
