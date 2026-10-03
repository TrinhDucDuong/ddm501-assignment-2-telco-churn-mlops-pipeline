"""Run with python -m churn --config config.yaml."""

import argparse
import json

from churn.config import load_config
from churn.mlops import STAGES, run_mlops


def main() -> None:
    """Resolve configuration and print the persisted run summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stage", choices=list(STAGES))
    args = parser.parse_args()
    config = load_config(args.config)
    result = STAGES[args.stage](config) if args.stage else run_mlops(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
