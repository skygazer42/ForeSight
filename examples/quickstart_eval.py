import json
from foresight import eval_model, forecast_model
from foresight.datasets.loaders import load_dataset
from foresight.datasets.registry import get_dataset_spec


def main() -> None:
    """
    Quickstart example: evaluate a few builtin models, then produce a direct future forecast
    from the end of the bundled `catfish` dataset.

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

    spec = get_dataset_spec(dataset)
    raw = load_dataset(dataset)
    future_df = forecast_model(
        model="theta",
        y=raw[y_col].to_numpy(),
        ds=raw[spec.time_col],
        horizon=horizon,
        model_params={"alpha": 0.2},
    )
    print("\n=== theta future forecast ===")
    print(future_df.to_string(index=False))


if __name__ == "__main__":
    main()
