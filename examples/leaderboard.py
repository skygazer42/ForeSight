from __future__ import annotations

from collections.abc import Iterable

from foresight.eval_forecast import eval_model
from foresight.models.registry import get_model_spec, list_models


def _markdown_table(rows: list[dict], cols: Iterable[str]) -> str:
    cols = list(cols)

    def _fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    """
    A tiny leaderboard over all "core" registered models (no optional extras).
    """
    dataset = "catfish"
    y_col = "Total"
    horizon = 3
    step = 3
    min_train_size = 12

    keys = [k for k in list_models() if not get_model_spec(k).requires]

    rows: list[dict] = []
    for key in keys:
        try:
            res = eval_model(
                model=key,
                dataset=dataset,
                y_col=y_col,
                horizon=horizon,
                step=step,
                min_train_size=min_train_size,
            )
        except Exception:
            continue

        rows.append(
            {
                "model": key,
                "mae": float(res["mae"]),
                "rmse": float(res["rmse"]),
                "mape": float(res["mape"]),
                "smape": float(res["smape"]),
                "notes": get_model_spec(key).description,
            }
        )

    rows.sort(key=lambda r: float(r.get("mae", float("inf"))))
    print(_markdown_table(rows, cols=["model", "mae", "rmse", "mape", "smape", "notes"]))


if __name__ == "__main__":
    main()
