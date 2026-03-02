import importlib.util

import numpy as np
import pytest

from foresight.models.registry import make_forecaster, make_global_forecaster


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_xformer_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 4.0) + 0.05 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-xformer-performer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))

    f2 = make_forecaster(
        "torch-xformer-probsparse-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        probsparse_top_u=16,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))

    f3 = make_forecaster(
        "torch-xformer-autocorr-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        autocorr_top_k=4,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat3 = f3(y, 5)
    assert yhat3.shape == (5,)
    assert np.all(np.isfinite(yhat3))

    f4 = make_forecaster(
        "torch-xformer-reformer-ln-gelu-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        reformer_bucket_size=8,
        reformer_n_hashes=1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat4 = f4(y, 5)
    assert yhat4.shape == (5,)
    assert np.all(np.isfinite(yhat4))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_seq2seq_local_smoke():
    y = np.sin(np.arange(120, dtype=float) / 5.0) + 0.03 * np.arange(120, dtype=float)
    f = make_forecaster(
        "torch-seq2seq-attn-lstm-direct",
        lags=24,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_mamba_rwkv_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 6.0) + 0.04 * np.arange(140, dtype=float)

    f1 = make_forecaster(
        "torch-mamba-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        conv_kernel=3,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-rwkv-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_hyena_local_smoke():
    y = np.sin(np.arange(140, dtype=float) / 7.0) + 0.04 * np.arange(140, dtype=float)
    f = make_forecaster(
        "torch-hyena-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        kernel_size=32,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat = f(y, 5)
    assert yhat.shape == (5,)
    assert np.all(np.isfinite(yhat))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_dilated_rnn_and_kan_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 9.0) + 0.03 * np.arange(160, dtype=float)

    f1 = make_forecaster(
        "torch-dilated-rnn-direct",
        lags=48,
        cell="gru",
        hidden_size=32,
        num_layers=2,
        dilation_base=2,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-kan-direct",
        lags=48,
        d_model=32,
        num_layers=1,
        grid_size=8,
        grid_range=2.0,
        dropout=0.0,
        linear_skip=True,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_scinet_and_etsformer_local_smoke():
    y = np.sin(np.arange(160, dtype=float) / 8.0) + 0.03 * np.arange(160, dtype=float)

    f1 = make_forecaster(
        "torch-scinet-direct",
        lags=48,
        d_model=32,
        num_stages=2,
        conv_kernel=5,
        ffn_dim=64,
        dropout=0.0,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat1 = f1(y, 5)
    assert yhat1.shape == (5,)
    assert np.all(np.isfinite(yhat1))

    f2 = make_forecaster(
        "torch-etsformer-direct",
        lags=48,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        alpha_init=0.3,
        beta_init=0.1,
        epochs=2,
        batch_size=16,
        patience=2,
        device="cpu",
        seed=0,
    )
    yhat2 = f2(y, 5)
    assert yhat2.shape == (5,)
    assert np.all(np.isfinite(yhat2))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_torch_xformer_and_rnn_global_smoke():
    import pandas as pd

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    cutoff = ds[-6]
    horizon = 5

    g1 = make_global_forecaster(
        "torch-xformer-performer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1 = g1(long_df, cutoff, horizon)
    assert set(pred1.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1["yhat"].to_numpy(dtype=float)))

    g1b = make_global_forecaster(
        "torch-xformer-probsparse-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        probsparse_top_u=16,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1b = g1b(long_df, cutoff, horizon)
    assert set(pred1b.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1b["yhat"].to_numpy(dtype=float)))

    g1c = make_global_forecaster(
        "torch-xformer-autocorr-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        autocorr_top_k=4,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1c = g1c(long_df, cutoff, horizon)
    assert set(pred1c.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1c["yhat"].to_numpy(dtype=float)))

    g1d = make_global_forecaster(
        "torch-xformer-reformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        reformer_bucket_size=8,
        reformer_n_hashes=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred1d = g1d(long_df, cutoff, horizon)
    assert set(pred1d.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred1d["yhat"].to_numpy(dtype=float)))

    g2 = make_global_forecaster(
        "torch-rnn-gru-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred2 = g2(long_df, cutoff, horizon)
    assert set(pred2.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred2["yhat"].to_numpy(dtype=float)))

    g3 = make_global_forecaster(
        "torch-patchtst-global",
        context_length=32,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred3 = g3(long_df, cutoff, horizon)
    assert set(pred3.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred3["yhat"].to_numpy(dtype=float)))

    g4 = make_global_forecaster(
        "torch-tsmixer-global",
        context_length=32,
        d_model=32,
        num_blocks=2,
        token_mixing_hidden=64,
        channel_mixing_hidden=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred4 = g4(long_df, cutoff, horizon)
    assert set(pred4.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred4["yhat"].to_numpy(dtype=float)))

    g5 = make_global_forecaster(
        "torch-itransformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred5 = g5(long_df, cutoff, horizon)
    assert set(pred5.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred5["yhat"].to_numpy(dtype=float)))

    g6 = make_global_forecaster(
        "torch-timesnet-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        top_k=2,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred6 = g6(long_df, cutoff, horizon)
    assert set(pred6.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred6["yhat"].to_numpy(dtype=float)))

    g7 = make_global_forecaster(
        "torch-seq2seq-attn-lstm-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        attention="bahdanau",
        teacher_forcing=0.6,
        teacher_forcing_final=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred7 = g7(long_df, cutoff, horizon)
    assert set(pred7.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred7["yhat"].to_numpy(dtype=float)))

    g8 = make_global_forecaster(
        "torch-tcn-global",
        context_length=32,
        channels=(32, 32),
        kernel_size=3,
        dilation_base=2,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred8 = g8(long_df, cutoff, horizon)
    assert set(pred8.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred8["yhat"].to_numpy(dtype=float)))

    g9 = make_global_forecaster(
        "torch-nbeats-global",
        context_length=32,
        num_blocks=2,
        num_layers=2,
        layer_width=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred9 = g9(long_df, cutoff, horizon)
    assert set(pred9.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred9["yhat"].to_numpy(dtype=float)))

    g10 = make_global_forecaster(
        "torch-nhits-global",
        context_length=32,
        pool_sizes=(1, 2),
        num_blocks=2,
        num_layers=2,
        layer_width=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred10 = g10(long_df, cutoff, horizon)
    assert set(pred10.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred10["yhat"].to_numpy(dtype=float)))

    g11 = make_global_forecaster(
        "torch-tide-global",
        context_length=32,
        d_model=32,
        hidden_size=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred11 = g11(long_df, cutoff, horizon)
    assert set(pred11.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred11["yhat"].to_numpy(dtype=float)))

    g12 = make_global_forecaster(
        "torch-wavenet-global",
        context_length=32,
        channels=16,
        num_layers=4,
        kernel_size=2,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred12 = g12(long_df, cutoff, horizon)
    assert set(pred12.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred12["yhat"].to_numpy(dtype=float)))

    g13 = make_global_forecaster(
        "torch-resnet1d-global",
        context_length=32,
        channels=16,
        num_blocks=2,
        kernel_size=3,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred13 = g13(long_df, cutoff, horizon)
    assert set(pred13.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred13["yhat"].to_numpy(dtype=float)))

    g14 = make_global_forecaster(
        "torch-inception-global",
        context_length=32,
        channels=16,
        num_blocks=2,
        kernel_sizes=(3, 5),
        bottleneck_channels=8,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred14 = g14(long_df, cutoff, horizon)
    assert set(pred14.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred14["yhat"].to_numpy(dtype=float)))

    g15 = make_global_forecaster(
        "torch-lstnet-global",
        context_length=32,
        cnn_channels=8,
        kernel_size=3,
        rnn_hidden=16,
        skip=8,
        highway_window=16,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred15 = g15(long_df, cutoff, horizon)
    assert set(pred15.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred15["yhat"].to_numpy(dtype=float)))

    g16 = make_global_forecaster(
        "torch-fnet-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred16 = g16(long_df, cutoff, horizon)
    assert set(pred16.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred16["yhat"].to_numpy(dtype=float)))

    g17 = make_global_forecaster(
        "torch-gmlp-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=32,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred17 = g17(long_df, cutoff, horizon)
    assert set(pred17.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred17["yhat"].to_numpy(dtype=float)))

    g18 = make_global_forecaster(
        "torch-ssm-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred18 = g18(long_df, cutoff, horizon)
    assert set(pred18.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred18["yhat"].to_numpy(dtype=float)))

    g19 = make_global_forecaster(
        "torch-transformer-encdec-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred19 = g19(long_df, cutoff, horizon)
    assert set(pred19.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred19["yhat"].to_numpy(dtype=float)))

    g20 = make_global_forecaster(
        "torch-nlinear-global",
        context_length=32,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred20 = g20(long_df, cutoff, horizon)
    assert set(pred20.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred20["yhat"].to_numpy(dtype=float)))

    g21 = make_global_forecaster(
        "torch-dlinear-global",
        context_length=32,
        ma_window=7,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred21 = g21(long_df, cutoff, horizon)
    assert set(pred21.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred21["yhat"].to_numpy(dtype=float)))

    g22 = make_global_forecaster(
        "torch-deepar-global",
        context_length=32,
        hidden_size=32,
        num_layers=1,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred22 = g22(long_df, cutoff, horizon)
    assert set(pred22.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred22["yhat"].to_numpy(dtype=float)))

    g23 = make_global_forecaster(
        "torch-fedformer-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        modes=8,
        ma_window=7,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred23 = g23(long_df, cutoff, horizon)
    assert set(pred23.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred23["yhat"].to_numpy(dtype=float)))

    g24 = make_global_forecaster(
        "torch-nonstationary-transformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        quantiles="0.1,0.5,0.9",
        device="cpu",
        seed=0,
    )
    pred24 = g24(long_df, cutoff, horizon)
    assert set(pred24.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred24["yhat"].to_numpy(dtype=float)))

    g25 = make_global_forecaster(
        "torch-mamba-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        conv_kernel=3,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred25 = g25(long_df, cutoff, horizon)
    assert set(pred25.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred25["yhat"].to_numpy(dtype=float)))

    g26 = make_global_forecaster(
        "torch-rwkv-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        dropout=0.0,
        quantiles="0.1,0.5,0.9",
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred26 = g26(long_df, cutoff, horizon)
    assert set(pred26.columns) >= {"unique_id", "ds", "yhat", "yhat_p10", "yhat_p50", "yhat_p90"}
    assert np.all(np.isfinite(pred26["yhat"].to_numpy(dtype=float)))

    g27 = make_global_forecaster(
        "torch-hyena-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        ffn_dim=64,
        kernel_size=32,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred27 = g27(long_df, cutoff, horizon)
    assert set(pred27.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred27["yhat"].to_numpy(dtype=float)))

    g28 = make_global_forecaster(
        "torch-dilated-rnn-global",
        context_length=32,
        cell="gru",
        d_model=32,
        num_layers=2,
        dilation_base=2,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred28 = g28(long_df, cutoff, horizon)
    assert set(pred28.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred28["yhat"].to_numpy(dtype=float)))

    g29 = make_global_forecaster(
        "torch-kan-global",
        context_length=32,
        d_model=32,
        num_layers=1,
        grid_size=8,
        grid_range=2.0,
        dropout=0.0,
        linear_skip=True,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred29 = g29(long_df, cutoff, horizon)
    assert set(pred29.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred29["yhat"].to_numpy(dtype=float)))

    g30 = make_global_forecaster(
        "torch-scinet-global",
        context_length=32,
        d_model=32,
        num_stages=2,
        conv_kernel=5,
        ffn_dim=64,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred30 = g30(long_df, cutoff, horizon)
    assert set(pred30.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred30["yhat"].to_numpy(dtype=float)))

    g31 = make_global_forecaster(
        "torch-etsformer-global",
        context_length=32,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        alpha_init=0.3,
        beta_init=0.1,
        dropout=0.0,
        sample_step=4,
        epochs=1,
        val_split=0.0,
        batch_size=32,
        patience=2,
        x_cols=("promo",),
        device="cpu",
        seed=0,
    )
    pred31 = g31(long_df, cutoff, horizon)
    assert set(pred31.columns) >= {"unique_id", "ds", "yhat"}
    assert np.all(np.isfinite(pred31["yhat"].to_numpy(dtype=float)))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_global_cv_preserves_quantile_columns():
    import pandas as pd

    from foresight.cv import cross_validation_predictions_long_df

    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    for uid, amp in [("s0", 1.0), ("s1", 1.2), ("s2", 0.8)]:
        base = amp * np.sin(np.arange(ds.size, dtype=float) / 6.0) + 0.01 * np.arange(ds.size)
        y = base + 0.05 * rng.standard_normal(ds.size)
        promo = (rng.random(ds.size) < 0.1).astype(float)
        for t, yv, pv in zip(ds, y, promo, strict=True):
            rows.append({"unique_id": uid, "ds": t, "y": float(yv), "promo": float(pv)})
    long_df = pd.DataFrame(rows)

    pred_df = cross_validation_predictions_long_df(
        model="torch-itransformer-global",
        long_df=long_df,
        horizon=3,
        step_size=3,
        min_train_size=40,
        model_params={
            "context_length": 32,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "sample_step": 4,
            "epochs": 1,
            "batch_size": 32,
            "patience": 2,
            "val_split": 0.0,
            "x_cols": ("promo",),
            "quantiles": "0.1,0.5,0.9",
            "device": "cpu",
            "seed": 0,
        },
        n_windows=1,
    )

    assert set(pred_df.columns) >= {"unique_id", "ds", "cutoff", "step", "y", "yhat", "yhat_p10"}
    assert np.all(np.isfinite(pred_df["yhat"].to_numpy(dtype=float)))
