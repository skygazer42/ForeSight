import sys
from pathlib import Path
import torch
import numpy as np

# Allow running this example without requiring an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from foresight.models.registry import make_forecaster


def main() -> None:
    """
    Demo: the RNN Paper Zoo + RNN Zoo (paper-named recurrent structures).

    Run:
      pip install -e ".[dev,torch]"
      python examples/rnn_paper_zoo.py
    """

    # Simple synthetic series
    rng = np.random.default_rng(0)
    t = np.arange(240, dtype=float)
    y = 0.02 * t + np.sin(2.0 * np.pi * t / 30.0) + 0.2 * rng.standard_normal(t.shape[0])

    horizon = 14
    common = {
        "lags": 48,
        "hidden_size": 32,
        "epochs": 10,
        "batch_size": 32,
        "lr": 1e-3,
        "seed": 0,
        "patience": 5,
        "device": "cpu",
    }

    models = [
        ("torch-rnnpaper-elman-srn-direct", common),
        ("torch-rnnpaper-peephole-lstm-direct", common),
        ("torch-rnnpaper-deep-ar-direct", {**common, "epochs": 20}),
        ("torch-rnnzoo-elman-attn-direct", common),
        ("torch-rnnzoo-phased-lstm-bidir-direct", common),
    ]

    for key, params in models:
        f = make_forecaster(key, **params)
        yhat = f(y, horizon)
        print(f"\n=== {key} ===")
        print(np.round(yhat[:10], 3))


if __name__ == "__main__":
    main()
