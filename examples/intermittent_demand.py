import numpy as np
from foresight.models.intermittent import adida_forecast, croston_classic_forecast, tsb_forecast


def main() -> None:
    """
    Example: intermittent demand baselines (Croston / TSB / ADIDA).
    """
    y = np.array([0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 2, 0], dtype=float)
    horizon = 6

    print("y:", y.tolist())
    print("\nCroston:", croston_classic_forecast(y, horizon, alpha=0.2).tolist())
    print("TSB:", tsb_forecast(y, horizon, alpha=0.2, beta=0.2).tolist())
    print("ADIDA:", adida_forecast(y, horizon, agg_period=3, base="ses", alpha=0.2).tolist())


if __name__ == "__main__":
    main()
