import numpy as np

from foresight.cv import cross_validation_predictions


def test_cv_predictions_catfish_shapes_and_columns():
    df = cross_validation_predictions(
        model="naive-last",
        dataset="catfish",
        y_col="Total",
        horizon=3,
        step_size=3,
        min_train_size=12,
    )
    assert set(df.columns) == {"unique_id", "ds", "cutoff", "step", "y", "yhat", "model"}
    assert (df["step"].min(), df["step"].max()) == (1, 3)

    # catfish has 324 rows -> windows: train_end = 12..321 step 3 => 104 windows; each yields horizon rows
    assert len(df) == 104 * 3

    # naive-last: within each cutoff, yhat repeats the last observed y at cutoff
    grouped = df.groupby("cutoff", sort=False)
    for _cutoff, g in list(grouped)[:3]:
        assert g["yhat"].nunique() == 1


def test_cv_predictions_max_train_size_changes_mean_model():
    df_expanding = cross_validation_predictions(
        model="mean",
        dataset="catfish",
        y_col="Total",
        horizon=1,
        step_size=10,
        min_train_size=12,
        max_train_size=None,
        n_windows=5,
    )
    df_rolling = cross_validation_predictions(
        model="mean",
        dataset="catfish",
        y_col="Total",
        horizon=1,
        step_size=10,
        min_train_size=12,
        max_train_size=12,
        n_windows=5,
    )
    assert len(df_expanding) == len(df_rolling)
    # Rolling mean over last 12 points should differ from expanding mean at least sometimes
    assert np.any(np.abs(df_expanding["yhat"].to_numpy() - df_rolling["yhat"].to_numpy()) > 1e-9)
