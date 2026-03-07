# ForeSight

ForeSight is a registry-driven time-series forecasting package with a stable
Python API, a CLI-first workflow, probabilistic forecasts, and a broad mix of
classical, ML, and deep-learning model families.

## Start here

- [Model capability matrix](models.md)
- [Python API reference](api.md)
- [Install guide](INSTALL.md)
- [Development guide](DEVELOPMENT.md)
- [Release guide](RELEASE.md)
- [RNN Paper Zoo](rnn_paper_zoo.md)
- [RNN Zoo](rnn_zoo.md)

## Stable entry points

The public import surface is the root package:

```python
from foresight import eval_model, forecast_model, make_forecaster_object
```

Use the generated model matrix for registry capabilities, and use the API page
for the supported forecast, evaluation, artifact, tuning, and data-prep entry
points.
