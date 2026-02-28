from __future__ import annotations

import json

from foresight.eval_forecast import eval_model


def main() -> None:
    """
    Quickstart example: evaluate a few builtin models on the bundled `catfish` dataset.

    Run from repo root after installing:
        pip install -e ".[dev]"
    """
    dataset = "catfish"
    y_col = "Total"
    horizon = 3
    step = 3
    min_train_size = 12

    experiments = [
        ("naive-last", {}),
        ("seasonal-naive", {"season_length": 12}),
        ("theta", {"alpha": 0.2}),
        ("moving-average", {"window": 3}),
    ]

    for model_key, model_params in experiments:
        res = eval_model(
            model=model_key,
            dataset=dataset,
            y_col=y_col,
            horizon=horizon,
            step=step,
            min_train_size=min_train_size,
            model_params=model_params,
        )
        print(f"\n=== {model_key} ===")
        print(json.dumps(res, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
