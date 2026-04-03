import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from foresight.models import make_forecaster_object, make_global_forecaster_object
from foresight.models.neural_runtime import (
    coerce_torch_train_config_params,
    summarize_model_runtime,
)
from foresight.models import torch_global, torch_nn, torch_rnn_paper_zoo, torch_seq2seq
from foresight.serialization import load_forecaster, load_forecaster_artifact, save_forecaster


def _small_panel_long_df_with_promo() -> tuple[pd.DataFrame, pd.Timestamp]:
    ds = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []
    for uid, bias in [("s0", 0.0), ("s1", 1.0)]:
        for i, d in enumerate(ds):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": d,
                    "y": float(bias + 0.1 * i),
                    "promo": float(i % 7 == 0),
                }
            )
    return pd.DataFrame(rows), ds[-6]


def test_global_torch_forecasters_use_shared_train_config_contract() -> None:
    assert torch_global.TorchGlobalTrainConfig is torch_nn.TorchTrainConfig


def test_coerce_torch_train_config_params_uses_shared_contract() -> None:
    coerced = coerce_torch_train_config_params(
        {
            "min_epochs": "2",
            "amp": "true",
            "amp_dtype": 123,
            "warmup_epochs": "3",
            "min_lr": "0.01",
            "num_workers": "4",
            "pin_memory": "yes",
            "persistent_workers": "on",
            "sam_adaptive": "1",
            "save_best_checkpoint": "true",
            "resume_checkpoint_strict": "false",
            "checkpoint_dir": 99,
            "tensorboard_log_dir": 123,
            "tensorboard_run_name": 456,
            "tensorboard_flush_secs": "15",
            "mlflow_tracking_uri": 789,
            "mlflow_experiment_name": 321,
            "mlflow_run_name": 654,
            "wandb_project": 111,
            "wandb_entity": 222,
            "wandb_run_name": 333,
            "wandb_dir": 444,
            "wandb_mode": 555,
            "device": "cuda",
            "unknown": "ignored",
        }
    )

    assert coerced == {
        "min_epochs": 2,
        "amp": True,
        "amp_dtype": "123",
        "warmup_epochs": 3,
        "min_lr": 0.01,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "sam_adaptive": True,
        "save_best_checkpoint": True,
        "resume_checkpoint_strict": False,
        "checkpoint_dir": "99",
        "tensorboard_log_dir": "123",
        "tensorboard_run_name": "456",
        "tensorboard_flush_secs": 15,
        "mlflow_tracking_uri": "789",
        "mlflow_experiment_name": "321",
        "mlflow_run_name": "654",
        "wandb_project": "111",
        "wandb_entity": "222",
        "wandb_run_name": "333",
        "wandb_dir": "444",
        "wandb_mode": "555",
    }


def test_summarize_model_runtime_exposes_structured_torch_training_sections() -> None:
    runtime = summarize_model_runtime(
        model_key="torch-mlp-direct",
        model_params={
            "device": "cpu",
            "epochs": 2,
            "batch_size": 16,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "grad_accum_steps": 2,
            "grad_clip_mode": "value",
            "grad_clip_norm": 0.0,
            "grad_clip_value": 0.1,
            "scheduler": "plateau",
            "scheduler_patience": 3,
            "scheduler_plateau_factor": 0.5,
            "scheduler_plateau_threshold": 0.0,
            "warmup_epochs": 1,
            "min_lr": 1e-5,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "min_delta": 0.01,
            "val_split": 0.25,
            "patience": 4,
            "min_epochs": 2,
            "restore_best": True,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "ema_decay": 0.995,
            "ema_warmup_epochs": 1,
            "swa_start_epoch": -1,
            "lookahead_steps": 5,
            "lookahead_alpha": 0.5,
            "sam_rho": 0.05,
            "sam_adaptive": True,
            "horizon_loss_decay": 0.8,
            "input_dropout": 0.1,
            "temporal_dropout": 0.2,
            "grad_noise_std": 0.01,
            "gc_mode": "all",
            "agc_clip_factor": 0.02,
            "agc_eps": 1e-3,
            "checkpoint_dir": "/tmp/checkpoints",
            "save_best_checkpoint": True,
            "save_last_checkpoint": True,
            "resume_checkpoint_path": "/tmp/resume.pt",
            "resume_checkpoint_strict": False,
            "tensorboard_log_dir": "/tmp/tensorboard",
            "tensorboard_run_name": "exp-7",
            "tensorboard_flush_secs": 15,
            "mlflow_tracking_uri": "file:///tmp/mlruns",
            "mlflow_experiment_name": "foresight-exp",
            "mlflow_run_name": "trial-7",
            "wandb_project": "foresight",
            "wandb_entity": "core",
            "wandb_run_name": "trial-7",
            "wandb_dir": "/tmp/wandb",
            "wandb_mode": "offline",
            "quantiles": (0.1, 0.5, 0.9),
        },
    )

    assert runtime is not None
    assert runtime["optimizer"]["name"] == "adamw"
    assert runtime["optimizer"]["grad_clip"]["mode"] == "value"
    assert runtime["scheduler"]["name"] == "plateau"
    assert runtime["scheduler"]["plateau_factor"] == 0.5
    assert runtime["monitor"]["metric"] == "val_loss"
    assert runtime["dataloader"]["num_workers"] == 2
    assert runtime["strategies"]["lookahead"]["steps"] == 5
    assert runtime["strategies"]["sam"]["rho"] == 0.05
    assert runtime["checkpoints"]["save_last"] is True
    assert runtime["checkpoints"]["resume_strict"] is False
    assert runtime["tracking"]["tensorboard"]["log_dir"] == "/tmp/tensorboard"
    assert runtime["tracking"]["tensorboard"]["run_name"] == "exp-7"
    assert runtime["tracking"]["tensorboard"]["flush_secs"] == 15
    assert runtime["tracking"]["mlflow"]["tracking_uri"] == "file:///tmp/mlruns"
    assert runtime["tracking"]["mlflow"]["experiment_name"] == "foresight-exp"
    assert runtime["tracking"]["mlflow"]["run_name"] == "trial-7"
    assert runtime["tracking"]["wandb"]["project"] == "foresight"
    assert runtime["tracking"]["wandb"]["entity"] == "core"
    assert runtime["tracking"]["wandb"]["run_name"] == "trial-7"
    assert runtime["tracking"]["wandb"]["directory"] == "/tmp/wandb"
    assert runtime["tracking"]["wandb"]["mode"] == "offline"
    assert runtime["prediction"]["mode"] == "quantile"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_generic_torch_trainer_supports_single_input_batches() -> None:
    torch = torch_nn._require_torch()

    class _SingleInputRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, xb):
            return self.linear(xb)

    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=1,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        restore_best=False,
    )
    X = torch.randn(12, 4)
    Y = torch.randn(12, 2)
    train_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X, Y),
        cfg=cfg,
        shuffle=True,
    )

    model = torch_nn._train_torch_model_with_loaders(
        _SingleInputRegressor(),
        train_loader,
        None,
        cfg=cfg,
        device="cpu",
    )
    pred = model(X[:2])
    assert pred.shape == (2, 2)
    assert torch.isfinite(pred).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_generic_torch_trainer_writes_tensorboard_metrics(monkeypatch, tmp_path) -> None:
    torch = torch_nn._require_torch()
    recorded: list[tuple] = []
    checkpoint_dir = tmp_path / "checkpoints"

    class _FakeWriter:
        def add_scalar(self, tag, scalar_value, global_step):
            recorded.append(("scalar", str(tag), float(scalar_value), int(global_step)))

        def add_text(self, tag, text_string, global_step=None):
            recorded.append(
                (
                    "text",
                    str(tag),
                    str(text_string),
                    (None if global_step is None else int(global_step)),
                )
            )

        def add_hparams(self, hparam_dict, metric_dict, run_name=None):
            recorded.append(
                (
                    "hparams",
                    dict(hparam_dict),
                    dict(metric_dict),
                    (None if run_name is None else str(run_name)),
                )
            )

        def flush(self):
            recorded.append(("flush",))

        def close(self):
            recorded.append(("close",))

    def _fake_open_torch_tensorboard_writer(*, cfg):
        assert cfg.tensorboard_log_dir == str(tmp_path)
        assert cfg.tensorboard_run_name == "demo-run"
        assert cfg.tensorboard_flush_secs == 15
        return _FakeWriter(), (tmp_path / "demo-run").as_posix()

    monkeypatch.setattr(
        torch_nn,
        "_open_torch_tensorboard_writer",
        _fake_open_torch_tensorboard_writer,
    )

    class _SingleInputRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, xb):
            return self.linear(xb)

    cfg = torch_nn.TorchTrainConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=2,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        restore_best=False,
        tensorboard_log_dir=str(tmp_path),
        tensorboard_run_name="demo-run",
        tensorboard_flush_secs=15,
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=True,
        save_last_checkpoint=True,
    )
    X = torch.randn(12, 4)
    Y = torch.randn(12, 2)
    train_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X, Y),
        cfg=cfg,
        shuffle=True,
    )

    model = torch_nn._train_torch_model_with_loaders(
        _SingleInputRegressor(),
        train_loader,
        None,
        cfg=cfg,
        device="cpu",
    )

    pred = model(X[:2])
    assert pred.shape == (2, 2)
    scalar_tags = {item[1] for item in recorded if item[0] == "scalar"}
    text_tags = {item[1] for item in recorded if item[0] == "text"}
    hparams_records = [item for item in recorded if item[0] == "hparams"]
    assert "train/loss" in scalar_tags
    assert "train/lr" in scalar_tags
    assert "system/avg_grad_norm" in scalar_tags
    assert "foresight/config" in text_tags
    assert "foresight/device" in text_tags
    assert "foresight/artifacts" in text_tags
    assert len(hparams_records) == 1
    _, hparams_payload, metric_payload, run_name = hparams_records[0]
    assert hparams_payload["batch_size"] == 4
    assert hparams_payload["optimizer"] == "adam"
    assert hparams_payload["tensorboard_run_name"] == "demo-run"
    assert metric_payload["train/epochs_ran"] == 2
    assert metric_payload["monitor/final_best"] > 0.0
    assert run_name == "demo-run"
    artifact_payload = next(
        item[2] for item in recorded if item[:2] == ("text", "foresight/artifacts")
    )
    artifact_info = json.loads(artifact_payload)
    assert artifact_info["checkpoint_dir"] == checkpoint_dir.as_posix()
    assert artifact_info["best_checkpoint_path"] == (checkpoint_dir / "best.pt").as_posix()
    assert artifact_info["last_checkpoint_path"] == (checkpoint_dir / "last.pt").as_posix()
    assert ("flush",) in recorded
    assert ("close",) in recorded


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_generic_torch_trainer_logs_mlflow_and_wandb_tracking(monkeypatch, tmp_path) -> None:
    torch = torch_nn._require_torch()
    recorded: list[tuple] = []
    checkpoint_dir = tmp_path / "checkpoints"
    wandb_run_ref: dict[str, Any] = {}

    class _FakeMlflowRunInfo:
        run_id = "mlflow-run-1"

    class _FakeMlflowRun:
        info = _FakeMlflowRunInfo()

    class _FakeMlflow:
        def __init__(self) -> None:
            self._tracking_uri = ""

        def set_tracking_uri(self, uri: str) -> None:
            self._tracking_uri = str(uri)
            recorded.append(("mlflow_set_tracking_uri", str(uri)))

        def get_tracking_uri(self) -> str:
            return self._tracking_uri

        def set_experiment(self, name: str) -> None:
            recorded.append(("mlflow_set_experiment", str(name)))

        def start_run(self, run_name: str | None = None) -> _FakeMlflowRun:
            recorded.append(("mlflow_start_run", None if run_name is None else str(run_name)))
            return _FakeMlflowRun()

        def log_params(self, params: dict[str, Any]) -> None:
            recorded.append(("mlflow_log_params", dict(params)))

        def log_metric(self, key: str, value: float, step: int | None = None) -> None:
            recorded.append(("mlflow_log_metric", str(key), float(value), step))

        def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
            recorded.append(("mlflow_log_metrics", dict(metrics), step))

        def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
            recorded.append(("mlflow_log_dict", str(artifact_file), dict(payload)))

        def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
            recorded.append(
                (
                    "mlflow_log_artifact",
                    Path(local_path).name,
                    None if artifact_path is None else str(artifact_path),
                )
            )

        def end_run(self) -> None:
            recorded.append(("mlflow_end_run",))

    class _FakeWandbConfig(dict):
        def update(self, data: dict[str, Any], allow_val_change: bool | None = None) -> None:
            super().update(dict(data))
            recorded.append(("wandb_config_update", dict(data), allow_val_change))

    class _FakeWandbArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = str(name)
            self.type = str(type)
            recorded.append(("wandb_artifact_init", self.name, self.type))

        def add_file(self, local_path: str, name: str | None = None) -> None:
            recorded.append(
                (
                    "wandb_artifact_add_file",
                    Path(local_path).name,
                    None if name is None else str(name),
                )
            )

    class _FakeWandbRun:
        def __init__(
            self,
            *,
            project: str,
            entity: str | None,
            name: str | None,
            directory: str | None,
            mode: str | None,
            config: dict[str, Any] | None,
        ) -> None:
            self.project = str(project)
            self.entity = None if entity is None else str(entity)
            self.name = None if name is None else str(name)
            self.dir = None if directory is None else str(directory)
            self.mode = None if mode is None else str(mode)
            self.id = "wandb-run-1"
            self.path = "/".join(
                part
                for part in (self.entity or "anonymous", self.project, self.name or self.id)
                if part
            )
            self.config = _FakeWandbConfig()
            self.summary: dict[str, Any] = {}
            if config:
                self.config.update(dict(config), allow_val_change=True)

        def log(self, payload: dict[str, Any], step: int | None = None) -> None:
            recorded.append(("wandb_log", dict(payload), step))

        def log_artifact(self, artifact: _FakeWandbArtifact) -> None:
            recorded.append(("wandb_log_artifact", artifact.name, artifact.type))

        def finish(self) -> None:
            recorded.append(("wandb_finish",))

    class _FakeWandb:
        Artifact = _FakeWandbArtifact

        def init(
            self,
            *,
            project: str,
            entity: str | None = None,
            name: str | None = None,
            dir: str | None = None,
            mode: str | None = None,
            config: dict[str, Any] | None = None,
        ) -> _FakeWandbRun:
            recorded.append(
                (
                    "wandb_init",
                    str(project),
                    (None if entity is None else str(entity)),
                    (None if name is None else str(name)),
                    (None if dir is None else str(dir)),
                    (None if mode is None else str(mode)),
                )
            )
            run = _FakeWandbRun(
                project=project,
                entity=entity,
                name=name,
                directory=dir,
                mode=mode,
                config=config,
            )
            wandb_run_ref["run"] = run
            return run

    monkeypatch.setattr(torch_nn, "_require_mlflow", lambda: _FakeMlflow(), raising=False)
    monkeypatch.setattr(torch_nn, "_require_wandb", lambda: _FakeWandb(), raising=False)

    class _SingleInputRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, xb):
            return self.linear(xb)

    cfg = torch_nn.TorchTrainConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=2,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        restore_best=False,
        checkpoint_dir=str(checkpoint_dir),
        save_best_checkpoint=True,
        save_last_checkpoint=True,
        mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
        mlflow_experiment_name="foresight-exp",
        mlflow_run_name="demo-mlflow",
        wandb_project="foresight-tests",
        wandb_entity="core",
        wandb_run_name="demo-wandb",
        wandb_dir=str(tmp_path / "wandb"),
        wandb_mode="offline",
    )
    X = torch.randn(12, 4)
    Y = torch.randn(12, 2)
    train_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X, Y),
        cfg=cfg,
        shuffle=True,
    )

    model = torch_nn._train_torch_model_with_loaders(
        _SingleInputRegressor(),
        train_loader,
        None,
        cfg=cfg,
        device="cpu",
    )

    pred = model(X[:2])
    assert pred.shape == (2, 2)

    mlflow_metric_keys = {item[1] for item in recorded if item[0] == "mlflow_log_metric"} | {
        key for item in recorded if item[0] == "mlflow_log_metrics" for key in item[1]
    }
    assert ("mlflow_set_experiment", "foresight-exp") in recorded
    assert ("mlflow_start_run", "demo-mlflow") in recorded
    assert "train/loss" in mlflow_metric_keys
    assert "train/epochs_ran" in mlflow_metric_keys
    mlflow_params = next(item[1] for item in recorded if item[0] == "mlflow_log_params")
    assert mlflow_params["batch_size"] == 4
    assert mlflow_params["wandb_project"] == "foresight-tests"
    mlflow_artifact_manifest = next(
        item[2]
        for item in recorded
        if item[0] == "mlflow_log_dict" and "best_checkpoint_path" in item[2]
    )
    assert (
        mlflow_artifact_manifest["best_checkpoint_path"] == (checkpoint_dir / "best.pt").as_posix()
    )
    assert (
        mlflow_artifact_manifest["last_checkpoint_path"] == (checkpoint_dir / "last.pt").as_posix()
    )
    assert {item[1] for item in recorded if item[0] == "mlflow_log_artifact"} >= {
        "best.pt",
        "last.pt",
    }
    assert ("mlflow_end_run",) in recorded

    wandb_run = wandb_run_ref["run"]
    assert (
        "wandb_init",
        "foresight-tests",
        "core",
        "demo-wandb",
        str(tmp_path / "wandb"),
        "offline",
    ) in recorded
    assert wandb_run.config["batch_size"] == 4
    assert wandb_run.config["mlflow_experiment_name"] == "foresight-exp"
    wandb_metric_keys = {key for item in recorded if item[0] == "wandb_log" for key in item[1]}
    assert "train/loss" in wandb_metric_keys
    assert "train/epochs_ran" in wandb_metric_keys
    assert "foresight/device" in wandb_run.summary
    assert "foresight/artifacts" in wandb_run.summary
    wandb_artifacts = wandb_run.summary["foresight/artifacts"]
    assert wandb_artifacts["best_checkpoint_path"] == (checkpoint_dir / "best.pt").as_posix()
    assert wandb_artifacts["last_checkpoint_path"] == (checkpoint_dir / "last.pt").as_posix()
    assert {item[1] for item in recorded if item[0] == "wandb_artifact_add_file"} >= {
        "best.pt",
        "last.pt",
    }
    assert any(item[0] == "wandb_log_artifact" for item in recorded)
    assert ("wandb_finish",) in recorded


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_generic_torch_trainer_supports_multi_input_batches() -> None:
    torch = torch_nn._require_torch()

    class _MultiInputRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(3, 2)
            self.linear = torch.nn.Linear(6, 2)

        def forward(self, xb, ids):
            emb = self.embed(ids)
            return self.linear(torch.cat([xb, emb], dim=1))

    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=1,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        restore_best=False,
    )
    X = torch.randn(12, 4)
    ids = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    Y = torch.randn(12, 2)
    train_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X, ids, Y),
        cfg=cfg,
        shuffle=True,
    )

    model = torch_nn._train_torch_model_with_loaders(
        _MultiInputRegressor(),
        train_loader,
        None,
        cfg=cfg,
        device="cpu",
    )
    pred = model(X[:2], ids[:2])
    assert pred.shape == (2, 2)
    assert torch.isfinite(pred).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_generic_torch_trainer_supports_custom_batch_predict_fn() -> None:
    torch = torch_nn._require_torch()

    class _TeacherForcingRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(3, 2)
            self.linear = torch.nn.Linear(6, 2)

        def forward(
            self,
            xb,
            ids,
            *,
            y_true,
            teacher_forcing_ratio: float,
        ):
            emb = self.embed(ids)
            base = self.linear(torch.cat([xb, emb], dim=1))
            if y_true is None:
                return base
            return teacher_forcing_ratio * y_true + (1.0 - teacher_forcing_ratio) * base

    cfg = torch_nn.TorchTrainConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=4,
        seed=0,
        patience=2,
        loss="mse",
        val_split=0.25,
        grad_clip_norm=0.0,
        optimizer="adam",
        scheduler="plateau",
        monitor="val_loss",
        scheduler_patience=1,
        scheduler_plateau_factor=0.5,
        scheduler_plateau_threshold=0.0,
        restore_best=False,
    )
    X = torch.randn(12, 4)
    ids = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    Y = torch.randn(12, 2)
    train_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X[:9], ids[:9], Y[:9]),
        cfg=cfg,
        shuffle=True,
    )
    val_loader = torch_nn._make_torch_dataloader(
        torch,
        torch.utils.data.TensorDataset(X[9:], ids[9:], Y[9:]),
        cfg=cfg,
        shuffle=False,
    )

    def _batch_predict_fn(model, model_inputs, target, *, epoch_idx: int, training: bool):
        xb, batch_ids = model_inputs
        ratio = 0.6 if training and epoch_idx == 0 else 0.0
        return model(
            xb,
            batch_ids,
            y_true=(target if training else None),
            teacher_forcing_ratio=ratio,
        )

    model = torch_nn._train_torch_model_with_loaders(
        _TeacherForcingRegressor(),
        train_loader,
        val_loader,
        cfg=cfg,
        device="cpu",
        batch_predict_fn=_batch_predict_fn,
    )
    pred = _batch_predict_fn(model, (X[:2], ids[:2]), Y[:2], epoch_idx=1, training=False)
    assert pred.shape == (2, 2)
    assert torch.isfinite(pred).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_seq2seq_training_delegates_to_shared_trainer(monkeypatch) -> None:
    torch = torch_nn._require_torch()

    class _TinySeq2Seq(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, xb, yb, teacher_forcing_ratio: float):
            horizon = int(yb.shape[1])
            base = xb[:, -1, 0].unsqueeze(1).repeat(1, horizon)
            return base + self.bias * (1.0 + float(teacher_forcing_ratio))

    cfg = torch_nn.TorchTrainConfig(
        epochs=1,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=2,
        seed=0,
        patience=1,
        loss="mse",
        val_split=0.0,
        grad_clip_norm=0.0,
        optimizer="adam",
        restore_best=False,
    )
    X = np.random.default_rng(0).standard_normal((6, 4, 1)).astype(np.float32)
    Y = np.random.default_rng(1).standard_normal((6, 2)).astype(np.float32)

    def _fake_shared_trainer(
        model,
        train_loader,
        val_loader,
        *,
        cfg,
        device,
        loss_fn_override=None,
        batch_predict_fn=None,
        optimizer_factory=None,
        scheduler_factory=None,
    ):
        assert train_loader is not None
        assert val_loader is None
        assert batch_predict_fn is not None
        assert optimizer_factory is torch_seq2seq._make_seq2seq_optimizer
        assert scheduler_factory is torch_seq2seq._make_seq2seq_scheduler
        raise RuntimeError("delegated-shared-trainer")

    monkeypatch.setattr(
        torch_seq2seq,
        "_train_torch_model_with_loaders",
        _fake_shared_trainer,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="delegated-shared-trainer"):
        torch_seq2seq._train_seq2seq(
            _TinySeq2Seq(),
            X,
            Y,
            cfg=cfg,
            device="cpu",
            teacher_forcing_start=0.6,
            teacher_forcing_final=0.0,
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_rnnpaper_seq2seq_delegates_to_shared_train_loop(monkeypatch) -> None:
    def _fake_train_loop(
        model,
        X,
        Y,
        *,
        cfg,
        device,
        loss_fn_override=None,
        batch_predict_fn=None,
    ):
        assert X.shape[1:] == (8, 1)
        assert Y.shape[1] == 2
        assert batch_predict_fn is not None
        raise RuntimeError("delegated-rnnpaper-shared-train-loop")

    monkeypatch.setattr(
        torch_rnn_paper_zoo,
        "_train_loop",
        _fake_train_loop,
    )

    series = np.sin(np.arange(56, dtype=float) / 6.0) + 0.02 * np.arange(56, dtype=float)
    with pytest.raises(RuntimeError, match="delegated-rnnpaper-shared-train-loop"):
        torch_rnn_paper_zoo.torch_rnnpaper_direct_forecast(
            series,
            2,
            paper="seq2seq",
            lags=8,
            hidden_size=8,
            attn_hidden=4,
            epochs=1,
            batch_size=8,
            patience=1,
            seed=0,
            device="cpu",
        )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_torch_global_seq2seq_delegates_to_shared_train_loop(monkeypatch) -> None:
    def _fake_train_loop_global(
        model,
        X,
        ids,
        Y,
        *,
        cfg,
        device,
        loss_fn_override=None,
        batch_predict_fn=None,
    ):
        assert X.ndim == 3
        assert ids.ndim == 1
        assert Y.ndim == 2
        assert batch_predict_fn is not None
        raise RuntimeError("delegated-global-shared-train-loop")

    monkeypatch.setattr(
        torch_global,
        "_train_loop_global",
        _fake_train_loop_global,
    )

    long_df, cutoff = _small_panel_long_df_with_promo()
    forecaster = torch_global.torch_seq2seq_global_forecaster(
        context_length=16,
        x_cols=("promo",),
        epochs=1,
        val_split=0.0,
        batch_size=8,
        patience=1,
        seed=0,
        device="cpu",
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
    )

    with pytest.raises(RuntimeError, match="delegated-global-shared-train-loop"):
        forecaster(long_df, cutoff, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_save_and_load_local_torch_forecaster_reports_runtime_summary(tmp_path) -> None:
    y = np.sin(np.arange(96, dtype=float) / 6.0) + 0.02 * np.arange(96, dtype=float)
    forecaster = make_forecaster_object(
        "torch-multidim-rnn-direct",
        lags=24,
        hidden_size=16,
        epochs=2,
        batch_size=16,
        val_split=0.25,
        optimizer="adamw",
        scheduler="plateau",
        scheduler_patience=1,
        scheduler_plateau_factor=0.5,
        scheduler_plateau_threshold=0.0,
        grad_clip_mode="value",
        grad_clip_value=0.1,
        lookahead_steps=1,
        lookahead_alpha=0.5,
        save_last_checkpoint=False,
        seed=0,
        device="cpu",
    ).fit(y)
    path = tmp_path / "torch-local.pkl"

    metadata = save_forecaster(forecaster, path)
    payload = load_forecaster_artifact(path)
    loaded = load_forecaster(path)

    runtime = payload["metadata"]["train_schema"]["runtime"]
    assert runtime == metadata["train_schema"]["runtime"]
    assert runtime["family"] == "torch"
    assert runtime["device"] == "cpu"
    assert runtime["training"]["epochs"] == 2
    assert runtime["training"]["batch_size"] == 16
    assert runtime["optimizer"]["name"] == "adamw"
    assert runtime["scheduler"]["name"] == "plateau"
    assert runtime["monitor"]["metric"] == "auto"
    assert runtime["strategies"]["lookahead"]["steps"] == 1
    assert runtime["dataloader"]["batch_size"] == 16
    pred = loaded.predict(3)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_save_and_load_global_torch_forecaster_reports_runtime_summary(tmp_path) -> None:
    long_df, cutoff = _small_panel_long_df_with_promo()
    forecaster = make_global_forecaster_object(
        "torch-rnn-gru-global",
        context_length=16,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=16,
        x_cols=("promo",),
        seed=0,
        patience=2,
        device="cpu",
    ).fit(long_df)
    path = tmp_path / "torch-global.pkl"

    metadata = save_forecaster(forecaster, path)
    payload = load_forecaster_artifact(path)
    loaded = load_forecaster(path)

    runtime = payload["metadata"]["train_schema"]["runtime"]
    assert runtime == metadata["train_schema"]["runtime"]
    assert runtime["family"] == "torch"
    assert runtime["device"] == "cpu"
    assert runtime["training"]["epochs"] == 1
    assert runtime["training"]["batch_size"] == 16

    pred = loaded.predict(cutoff, 3)
    expected = forecaster.predict(cutoff, 3)
    assert pred.shape == expected.shape
    assert pred["unique_id"].tolist() == expected["unique_id"].tolist()
    assert pred["ds"].tolist() == expected["ds"].tolist()
    assert np.allclose(pred["yhat"].to_numpy(dtype=float), expected["yhat"].to_numpy(dtype=float))
