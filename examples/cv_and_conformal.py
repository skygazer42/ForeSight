import json
from foresight.conformal import apply_conformal_intervals, fit_conformal_intervals
from foresight.cv import cross_validation_predictions
from foresight.eval_predictions import evaluate_predictions
from foresight.metrics import interval_coverage, mean_interval_width


def main() -> None:
    """
    Example: run rolling-origin CV, fit conformal intervals from residuals,
    then compute simple point + interval diagnostics.
    """
    df = cross_validation_predictions(
        model="theta",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
        n_windows=30,
        model_params={"alpha": 0.2},
    )

    point = evaluate_predictions(df)
    print("=== Point metrics ===")
    print(json.dumps(point, ensure_ascii=False, indent=2, sort_keys=True))

    conf = fit_conformal_intervals(df, levels=(0.8, 0.9), per_step=True)
    df_i = apply_conformal_intervals(df, conf)

    for lv in conf.levels:
        p = int(round(lv * 100))
        cov = interval_coverage(df_i["y"], df_i[f"yhat_lo_{p}"], df_i[f"yhat_hi_{p}"])
        wid = mean_interval_width(df_i[f"yhat_lo_{p}"], df_i[f"yhat_hi_{p}"])
        print(f"\n=== Conformal {p}% ===")
        print(f"coverage={cov:.3f} mean_width={wid:.6g}")


if __name__ == "__main__":
    main()
