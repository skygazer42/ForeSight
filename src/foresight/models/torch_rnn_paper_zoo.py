from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .torch_nn import (
    TorchTrainConfig,
    _as_1d_float_array,
    _make_lagged_xy_multi,
    _normalize_series,
    _require_torch,
    _train_loop,
)

_IN_DIM_MIN_MSG = "in_dim must be >= 1"


@dataclass(frozen=True)
class RNNPaperModelSpec:
    key: str
    paper_id: str
    description: str


_PAPER_DEFS: list[tuple[str, str]] = [
    ("elman-srn", "Simple Recurrent Network / Elman RNN (Elman, 1990)"),
    ("jordan-srn", "Jordan recurrent network (Jordan, 1986)"),
    ("bidirectional-rnn", "Bidirectional RNN (Schuster & Paliwal, 1997)"),
    ("multi-dimensional-rnn", "Multi-Dimensional RNN (Graves et al., 2007)"),
    ("gated-feedback-rnn", "Gated Feedback RNN (Chung et al., 2015)"),
    ("hierarchical-multiscale-rnn", "Hierarchical Multiscale RNN (Chung et al., 2016)"),
    ("clockwork-rnn", "Clockwork RNN (Koutn铆k et al., 2014)"),
    ("dilated-rnn", "Dilated RNN (Chang et al., 2017)"),
    ("skip-rnn", "Skip RNN (Campos et al., 2017)"),
    ("sliced-rnn", "Sliced RNN (Yu & Liu, 2018)"),
    ("lstm", "LSTM (Hochreiter & Schmidhuber, 1997)"),
    ("forget-gate-lstm", "Forget-gate LSTM (Gers et al., 2000)"),
    ("peephole-lstm", "Peephole LSTM (Gers et al., 2002)"),
    ("lstm-projection", "LSTM with projection / LSTMP (Sak et al., 2014)"),
    ("cifg-lstm", "Coupled input-forget gate LSTM / CIFG (Greff et al., 2015)"),
    ("chrono-lstm", "Chrono LSTM (Tallec & Ollivier, 2018)"),
    ("phased-lstm", "Phased LSTM (Neil et al., 2016)"),
    ("grid-lstm", "Grid LSTM (Kalchbrenner et al., 2015)"),
    ("tree-lstm", "Tree-LSTM (Tai et al., 2015)"),
    ("nested-lstm", "Nested LSTM (Moniz & Krueger, 2017)"),
    ("on-lstm", "Ordered Neurons LSTM / ON-LSTM (Shen et al., 2018)"),
    ("lattice-rnn", "Lattice RNN (Ladhak et al., 2016)"),
    ("lattice-lstm", "Lattice LSTM (Zhang & Yang, 2018)"),
    ("wmc-lstm", "Working Memory Connections for LSTM (Landi et al., 2021)"),
    ("gru", "GRU (Cho et al., 2014)"),
    ("gru-variant-1", "GRU gate variant 1 (Dey & Salem, 2017)"),
    ("gru-variant-2", "GRU gate variant 2 (Dey & Salem, 2017)"),
    ("gru-variant-3", "GRU gate variant 3 (Dey & Salem, 2017)"),
    ("mgu", "Minimal Gated Unit / MGU (Zhou et al., 2016)"),
    ("mgu1", "MGU simplified variant 1 (Heck & Salem, 2017)"),
    ("mgu2", "MGU simplified variant 2 (Heck & Salem, 2017)"),
    ("mgu3", "MGU simplified variant 3 (Heck & Salem, 2017)"),
    ("ligru", "Light GRU / Li-GRU (Ravanelli et al., 2018)"),
    ("sru", "Simple Recurrent Unit / SRU (Lei et al., 2018)"),
    ("qrnn", "Quasi-Recurrent Neural Network / QRNN (Bradbury et al., 2016)"),
    ("indrnn", "IndRNN (Li et al., 2018)"),
    ("minimalrnn", "MinimalRNN (Chen, 2017)"),
    ("cfn", "Chaos-Free Network (Laurent & von Brecht, 2016)"),
    ("ran", "Recurrent Additive Network / RAN (Lee et al., 2017)"),
    ("atr", "Addition-Subtraction Twin-Gated RNN / ATR (Zhang et al., 2018)"),
    ("mut1", "MUT1 cell (J贸zefowicz et al., 2015)"),
    ("mut2", "MUT2 cell (J贸zefowicz et al., 2015)"),
    ("mut3", "MUT3 cell (J贸zefowicz et al., 2015)"),
    ("fast-rnn", "FastRNN (Kusupati et al., 2018)"),
    ("fast-grnn", "FastGRNN (Kusupati et al., 2018)"),
    ("fru", "Fourier Recurrent Unit / FRU (Zhang et al., 2018)"),
    ("rwa", "Recurrent Weighted Average / RWA (Ostmeyer & Cowell, 2017)"),
    ("rhn", "Recurrent Highway Network / RHN (Zilly et al., 2017)"),
    ("scrn", "Structurally Constrained RNN / SCRN (Mikolov et al., 2014)"),
    ("antisymmetric-rnn", "AntisymmetricRNN (Chang et al., 2019)"),
    ("cornn", "Coupled Oscillatory RNN / coRNN (Rusch & Mishra, 2020)"),
    ("unicornn", "UnICORNN (Rusch & Mishra, 2021)"),
    ("lem", "Long Expressive Memory / LEM (Rusch et al., 2021)"),
    ("tau-gru", "Weighted time-delay feedback GRU / 蟿-GRU (Erichson et al., 2022)"),
    ("dg-rnn", "Dynamic Gated RNN (Cheng et al., 2024)"),
    ("star", "Stackable recurrent cell / STAR (Turkoglu et al., 2019)"),
    ("strongly-typed-rnn", "Strongly-Typed RNN (Balduzzi & Ghifary, 2016)"),
    ("multiplicative-lstm", "Multiplicative LSTM / mLSTM (Krause et al., 2016)"),
    ("brc", "Bistable Recurrent Cell / BRC (Vecoven et al., 2021)"),
    ("nbrc", "Neuromodulated BRC / nBRC (Vecoven et al., 2021)"),
    ("residual-rnn", "Residual RNN (Yue et al., 2018)"),
    ("unitary-rnn", "Unitary RNN (Arjovsky et al., 2016)"),
    ("orthogonal-rnn", "Orthogonal RNN (Henaff et al., 2016)"),
    ("eunn", "Efficient Unitary Neural Network / EUNN (Jing et al., 2017)"),
    ("goru", "Gated Orthogonal RNN / GORU (Jing et al., 2017)"),
    ("ode-rnn", "ODE-RNN (Rubanova et al., 2019)"),
    ("neural-cde", "Neural CDE (Kidger et al., 2020)"),
    ("echo-state-network", "Echo State Network / ESN (Jaeger, 2001)"),
    ("deep-esn", "Deep Echo State Network (Gallicchio & Micheli, 2017)"),
    ("liquid-state-machine", "Liquid State Machine / LSM (Maass et al., 2002)"),
    ("conceptor-rnn", "Conceptor RNN (Jaeger, 2014)"),
    ("deep-ar", "DeepAR (Salinas et al., 2017)"),
    ("mqrnn", "MQRNN (Wen et al., 2017)"),
    ("deepstate", "DeepState (Rangapuram et al., 2018)"),
    ("lstnet", "LSTNet (Lai et al., 2018)"),
    ("esrnn", "ESRNN (Smyl, 2020)"),
    ("neural-turing-machine", "Neural Turing Machine / NTM (Graves et al., 2014)"),
    (
        "differentiable-neural-computer",
        "Differentiable Neural Computer / DNC (Graves et al., 2016)",
    ),
    ("memory-networks", "Memory Networks (Weston et al., 2014)"),
    ("end-to-end-memory-networks", "End-to-End Memory Networks (Sukhbaatar et al., 2015)"),
    ("dynamic-memory-networks", "Dynamic Memory Networks / DMN (Kumar et al., 2015)"),
    ("pointer-network", "Pointer Networks (Vinyals et al., 2015)"),
    ("pointer-sentinel-mixture", "Pointer Sentinel Mixture Model (Merity et al., 2017)"),
    ("copynet", "CopyNet (Gu et al., 2016)"),
    ("rnn-transducer", "RNN Transducer / RNN-T (Graves, 2012)"),
    ("seq2seq", "Seq2Seq (Sutskever et al., 2014)"),
    ("rnn-encoder-decoder", "RNN Encoder鈥揇ecoder (Cho et al., 2014)"),
    ("bahdanau-attention", "Additive (Bahdanau) attention (Bahdanau et al., 2015)"),
    ("luong-attention", "Multiplicative (Luong) attention (Luong et al., 2015)"),
    ("neural-stack", "Neural Stack (Grefenstette et al., 2015)"),
    ("neural-queue", "Neural Queue (Grefenstette et al., 2015)"),
    ("neural-ram", "Neural RAM / NRAM (Kurach et al., 2015)"),
    ("recurrent-attention-model", "Recurrent attention model (Mnih et al., 2014)"),
    ("convlstm", "ConvLSTM (Shi et al., 2015)"),
    ("convgru", "ConvGRU (Ballas et al., 2015)"),
    ("trajgru", "TrajGRU (Shi et al., 2017)"),
    ("predrnn", "PredRNN (Wang et al., 2017)"),
    ("predrnn-plus-plus", "PredRNN++ (Wang et al., 2018)"),
    ("dcrnn", "DCRNN (Li et al., 2018)"),
    ("structural-rnn", "Structural-RNN (Jain et al., 2016)"),
]

_PAPER_IDS = [paper_id for paper_id, _desc in _PAPER_DEFS]
_PAPER_DESC = {paper_id: desc for paper_id, desc in _PAPER_DEFS}


def list_rnnpaper_specs() -> list[RNNPaperModelSpec]:
    """
    Return the 100 paper-named RNN model specs registered under `torch-rnnpaper-*`.
    """
    out: list[RNNPaperModelSpec] = []
    for paper_id in _PAPER_IDS:
        out.append(
            RNNPaperModelSpec(
                key=f"torch-rnnpaper-{paper_id}-direct",
                paper_id=paper_id,
                description=_PAPER_DESC[paper_id],
            )
        )
    return out


def torch_rnnpaper_direct_forecast(
    train: Any,
    horizon: int,
    *,
    paper: str,
    lags: int = 24,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    # generic knobs used by some architectures
    attn_hidden: int = 32,
    kernel_size: int = 3,
    hops: int = 2,
    memory_slots: int = 16,
    memory_dim: int = 32,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    # training
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    seed: int = 0,
    normalize: bool = True,
    device: str = "cpu",
    patience: int = 10,
    loss: str = "mse",
    val_split: float = 0.0,
    grad_clip_norm: float = 0.0,
    optimizer: str = "adam",
    momentum: float = 0.9,
    scheduler: str = "none",
    scheduler_step_size: int = 10,
    scheduler_gamma: float = 0.1,
    restore_best: bool = True,
) -> np.ndarray:
    """
    Torch RNN Paper Zoo: 100 paper-named recurrent architectures with a unified
    lag-window direct forecasting interface.

    Notes:
    - Implementations are **lite** and optimized for fast baseline comparisons.
    - All models follow (train_1d, horizon) -> yhat[horizon] and train per-series.
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    lag_count = int(lags)
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    hid = int(hidden_size)
    if hid <= 0:
        raise ValueError("hidden_size must be >= 1")

    layers = int(num_layers)
    if layers <= 0:
        raise ValueError("num_layers must be >= 1")

    drop = float(dropout)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")

    paper_id = str(paper).strip().lower()
    if paper_id not in _PAPER_DESC:
        raise ValueError(f"Unknown rnnpaper architecture: {paper_id!r}")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    X_seq = X.reshape(X.shape[0], X.shape[1], 1)

    class _SeqEncoder(nn.Module):
        def forward(self, xb: Any) -> Any:
            raise NotImplementedError

    class _ScanCell(nn.Module):
        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            raise NotImplementedError

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            raise NotImplementedError

    class _ScanEncoder(_SeqEncoder):
        def __init__(self, cell: _ScanCell) -> None:
            super().__init__()
            self.cell = cell

        def forward(self, xb: Any) -> Any:  # (B, T, 1)
            B, T, _ = xb.shape
            state = self.cell.init_state(int(B), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                state, h_t = self.cell.step(xb[:, t, :], state, t=t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _StackedScanEncoder(_SeqEncoder):
        """
        Stack arbitrary scan-cells (manual recurrent cores) without relying on
        PyTorch's built-in recurrent modules.

        Dropout is applied between layers (not across time) when training.
        """

        def __init__(self, cells: list[_ScanCell], *, dropout: float = 0.0) -> None:
            super().__init__()
            if not cells:
                raise ValueError("cells must be non-empty")
            self.cells = nn.ModuleList(cells)
            self.dropout = float(dropout)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            states = [
                cell.init_state(int(B), device=xb.device, dtype=xb.dtype) for cell in self.cells
            ]
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                for i, cell in enumerate(self.cells):
                    states[i], h_t = cell.step(x_t, states[i], t=t)
                    x_t = h_t
                    if i < len(self.cells) - 1 and self.dropout > 0.0 and self.training:
                        x_t = F.dropout(x_t, p=self.dropout, training=True)
                outs.append(x_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _GRUCell(_ScanCell):
        def __init__(self, *, in_dim: int) -> None:
            super().__init__()
            d = int(in_dim)
            if d <= 0:
                raise ValueError(_IN_DIM_MIN_MSG)
            self.x2h = nn.Linear(d, 3 * hid, bias=True)
            self.h2h = nn.Linear(hid, 3 * hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            gi = self.x2h(x_t)
            gh = self.h2h(h_prev)
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h_t = (1.0 - z) * n + z * h_prev
            return h_t, h_t

    class _LSTMCell(_ScanCell):
        def __init__(
            self,
            *,
            in_dim: int,
            forget_bias_init: float | None = None,
            input_bias_init: float | None = None,
        ) -> None:
            super().__init__()
            d = int(in_dim)
            if d <= 0:
                raise ValueError(_IN_DIM_MIN_MSG)
            self.x2h = nn.Linear(d, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)
            if forget_bias_init is not None or input_bias_init is not None:
                with torch.no_grad():
                    if input_bias_init is not None:
                        self.x2h.bias[0:hid].fill_(float(input_bias_init))
                    if forget_bias_init is not None:
                        self.x2h.bias[hid : 2 * hid].fill_(float(forget_bias_init))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, c0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_prev + i * g
            h_t = o * torch.tanh(c_t)
            return (h_t, c_t), h_t

    class _LSTMPCell(_ScanCell):
        def __init__(self, *, in_dim: int, proj_dim: int) -> None:
            super().__init__()
            d = int(in_dim)
            p = int(proj_dim)
            if d <= 0:
                raise ValueError(_IN_DIM_MIN_MSG)
            if p <= 0:
                raise ValueError("proj_dim must be >= 1")
            self.proj_dim = p
            self.x2h = nn.Linear(d, 4 * hid, bias=True)
            self.h2h = nn.Linear(p, 4 * hid, bias=False)
            self.proj = nn.Linear(hid, p, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, self.proj_dim), device=device, dtype=dtype)
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, c0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c_t = f * c_prev + i * g
            m_t = o * torch.tanh(c_t)
            h_t = self.proj(m_t)
            return (h_t, c_t), h_t

    class _RNNCell(_ScanCell):
        def __init__(self, *, in_dim: int = 1) -> None:
            super().__init__()
            in_features = int(in_dim)
            if in_features <= 0:
                raise ValueError(_IN_DIM_MIN_MSG)
            self.x2h = nn.Linear(in_features, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_prev))
            return h_t, h_t

    class _JordanCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.y2h = nn.Linear(1, hid, bias=False)
            self.h2y = nn.Linear(hid, 1, bias=True)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            y0 = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            return (h0, y0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, y_prev = state
            h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_prev) + self.y2h(y_prev))
            y_t = self.h2y(h_t)
            return (h_t, y_t), h_t

    class _GRUVariantCell(_ScanCell):
        def __init__(self, *, variant: int) -> None:
            super().__init__()
            v = int(variant)
            if v not in {0, 1, 2, 3}:
                raise ValueError("GRU variant must be 0..3")
            self.variant = v
            # shared candidate
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            # gates
            if v == 3:
                self.z_bias = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
                self.r_bias = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
                self.x2z = None
                self.h2z = None
                self.x2r = None
                self.h2r = None
            else:
                self.z_bias = None
                self.r_bias = None
                self.x2z = nn.Linear(1, hid, bias=True)
                self.h2z = nn.Linear(hid, hid, bias=False)
                self.x2r = nn.Linear(1, hid, bias=True)
                self.h2r = nn.Linear(hid, hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            v = self.variant
            if v == 3:
                z = torch.sigmoid(self.z_bias).unsqueeze(0).expand_as(h_prev)
                r = torch.sigmoid(self.r_bias).unsqueeze(0).expand_as(h_prev)
            else:
                assert self.x2z is not None and self.h2z is not None
                assert self.x2r is not None and self.h2r is not None
                if v == 1:
                    z = torch.sigmoid(self.h2z(h_prev))
                    r = torch.sigmoid(self.h2r(h_prev))
                elif v == 2:
                    z = torch.sigmoid(self.x2z(x_t))
                    r = torch.sigmoid(self.x2r(x_t))
                else:
                    z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_prev))
                    r = torch.sigmoid(self.x2r(x_t) + self.h2r(h_prev))

            h_hat = torch.tanh(self.x2h(x_t) + self.h2h(r * h_prev))
            h_t = (1.0 - z) * h_prev + z * h_hat
            return h_t, h_t

    class _MGUCell(_ScanCell):
        def __init__(self, *, gate_mode: str = "full") -> None:
            super().__init__()
            self.gate_mode = str(gate_mode).lower().strip()
            self.x2f = nn.Linear(1, hid, bias=True)
            self.h2f = nn.Linear(hid, hid, bias=False)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.f_bias = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            mode = self.gate_mode
            if mode == "h-only":
                f = torch.sigmoid(self.h2f(h_prev) + self.f_bias)
            elif mode == "x-only":
                f = torch.sigmoid(self.x2f(x_t) + self.f_bias)
            elif mode == "bias-only":
                f = torch.sigmoid(self.f_bias).unsqueeze(0).expand_as(h_prev)
            else:
                f = torch.sigmoid(self.x2f(x_t) + self.h2f(h_prev) + self.f_bias)
            cand = torch.tanh(self.x2h(x_t) + self.h2h(f * h_prev))
            h_t = (1.0 - f) * h_prev + f * cand
            return h_t, h_t

    class _SRUCell(_ScanCell):
        """
        SRU-lite: cheap recurrence with two gates.
        """

        def __init__(self) -> None:
            super().__init__()
            self.x2u = nn.Linear(1, hid, bias=True)
            self.x2f = nn.Linear(1, hid, bias=True)
            self.x2r = nn.Linear(1, hid, bias=True)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return c0

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            c_prev = state
            u = torch.tanh(self.x2u(x_t))
            f_gate = torch.sigmoid(self.x2f(x_t))
            r_gate = torch.sigmoid(self.x2r(x_t))
            c_t = f_gate * c_prev + (1.0 - f_gate) * u
            h_t = r_gate * torch.tanh(c_t) + (1.0 - r_gate) * u
            return c_t, h_t

    class _PeepholeLSTMCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)
            self.w_ci = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
            self.w_cf = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
            self.w_co = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, c0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            i, f, g, o = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i + c_prev * self.w_ci)
            f = torch.sigmoid(f + c_prev * self.w_cf)
            g = torch.tanh(g)
            c_t = f * c_prev + i * g
            o = torch.sigmoid(o + c_t * self.w_co)
            h_t = o * torch.tanh(c_t)
            return (h_t, c_t), h_t

    class _CIFGLSTMCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 3 * hid, bias=True)
            self.h2h = nn.Linear(hid, 3 * hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, c0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, c_prev = state
            gates = self.x2h(x_t) + self.h2h(h_prev)
            f, g, o = gates.chunk(3, dim=-1)
            f = torch.sigmoid(f)
            i = 1.0 - f
            g = torch.tanh(g)
            c_t = f * c_prev + i * g
            o = torch.sigmoid(o)
            h_t = o * torch.tanh(c_t)
            return (h_t, c_t), h_t

    class _IndRNNCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.u = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            pre = self.x2h(x_t) + h_prev * self.u
            h_t = torch.relu(pre)
            return h_t, h_t

    class _MinimalRNNCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2z = nn.Linear(1, hid, bias=True)
            self.x2u = nn.Linear(1, hid, bias=True)
            self.h2u = nn.Linear(hid, hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            z = torch.tanh(self.x2z(x_t))
            u = torch.sigmoid(self.x2u(x_t) + self.h2u(h_prev))
            h_t = (1.0 - u) * h_prev + u * z
            return h_t, h_t

    class _RANCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2f = nn.Linear(1, hid, bias=True)
            self.h2f = nn.Linear(hid, hid, bias=False)
            self.x2i = nn.Linear(1, hid, bias=True)
            self.h2i = nn.Linear(hid, hid, bias=False)
            self.x2c = nn.Linear(1, hid, bias=True)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            c0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, c0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, c_prev = state
            f_gate = torch.sigmoid(self.x2f(x_t) + self.h2f(h_prev))
            i_gate = torch.sigmoid(self.x2i(x_t) + self.h2i(h_prev))
            x_proj = self.x2c(x_t)
            c_t = f_gate * c_prev + i_gate * x_proj
            h_t = torch.tanh(c_t)
            return (h_t, c_t), h_t

    class _FastRNNCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)
            h_hat = torch.tanh(self.x2h(x_t) + self.h2h(h_prev))
            h_t = alpha * h_prev + beta * h_hat
            return h_t, h_t

    class _FastGRNNCell(_ScanCell):
        def __init__(self) -> None:
            super().__init__()
            self.x2z = nn.Linear(1, hid, bias=True)
            self.h2z = nn.Linear(hid, hid, bias=False)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.x2r = nn.Linear(1, hid, bias=True)
            self.h2r = nn.Linear(hid, hid, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.beta_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            alpha = torch.sigmoid(self.alpha_logit)
            beta = torch.sigmoid(self.beta_logit)
            z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_prev))
            r = torch.sigmoid(self.x2r(x_t) + self.h2r(h_prev))
            h_hat = torch.tanh(self.x2h(x_t) + self.h2h(r * h_prev))
            h_t = (alpha * (1.0 - z) + beta) * h_prev + z * h_hat
            return h_t, h_t

    class _MUTCell(_ScanCell):
        def __init__(self, *, variant: int) -> None:
            super().__init__()
            v = int(variant)
            if v not in {1, 2, 3}:
                raise ValueError("mut variant must be 1..3")
            self.variant = v
            self.w1x = nn.Linear(1, hid, bias=True)
            self.w2h = nn.Linear(hid, hid, bias=False)
            if v == 1:
                self.w3x = nn.Linear(1, hid, bias=False)
                self.w4h = nn.Linear(hid, hid, bias=False)
                self.w3h = None
            elif v == 2:
                self.w3x = nn.Linear(1, hid, bias=False)
                self.w4h = None
                self.w3h = None
            else:
                self.w3x = None
                self.w4h = None
                self.w3h = nn.Linear(hid, hid, bias=True)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            a = self.w1x(x_t) * self.w2h(h_prev)
            if self.variant == 1:
                assert self.w3x is not None and self.w4h is not None
                b = self.w3x(x_t) + self.w4h(h_prev)
            elif self.variant == 2:
                assert self.w3x is not None
                b = self.w3x(x_t)
            else:
                assert self.w3h is not None
                b = self.w3h(h_prev)
            h_t = torch.tanh(a + b)
            return h_t, h_t

    class _SCRNCell(_ScanCell):
        def __init__(self, *, alpha: float = 0.95) -> None:
            super().__init__()
            self.alpha = float(alpha)
            self.x2s = nn.Linear(1, hid, bias=True)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.s2h = nn.Linear(hid, hid, bias=False)

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            h0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            s0 = torch.zeros((batch_size, hid), device=device, dtype=dtype)
            return (h0, s0)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev, s_prev = state
            a = float(self.alpha)
            s_t = (1.0 - a) * s_prev + a * self.x2s(x_t)
            h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_prev) + self.s2h(s_t))
            return (h_t, s_t), h_t

    class _RHNCell(_ScanCell):
        def __init__(self, *, depth: int) -> None:
            super().__init__()
            d = int(depth)
            if d <= 0:
                raise ValueError("rhn depth must be >= 1")
            self.depth = d
            self.x2h = nn.ModuleList(
                [nn.Linear(1, hid, bias=True)] + [nn.Linear(1, hid) for _ in range(d - 1)]
            )
            self.h2h = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(d)])
            self.x2t = nn.ModuleList(
                [nn.Linear(1, hid, bias=True)] + [nn.Linear(1, hid) for _ in range(d - 1)]
            )
            self.h2t = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(d)])

        def init_state(self, batch_size: int, *, device: Any, dtype: Any) -> Any:
            return torch.zeros((batch_size, hid), device=device, dtype=dtype)

        def step(self, x_t: Any, state: Any, *, t: int) -> tuple[Any, Any]:
            _t = t
            h_prev = state
            h_in = h_prev
            for i in range(self.depth):
                xi = x_t
                h_cand = torch.tanh(self.x2h[i](xi) + self.h2h[i](h_in))
                t_gate = torch.sigmoid(self.x2t[i](xi) + self.h2t[i](h_in))
                h_in = (1.0 - t_gate) * h_in + t_gate * h_cand
            return h_in, h_in

    class _ClockworkEncoder(_SeqEncoder):
        def __init__(self, *, periods: tuple[int, ...] = (1, 2, 4, 8)) -> None:
            super().__init__()
            ps = tuple(int(p) for p in periods)
            if not ps or any(p <= 0 for p in ps):
                raise ValueError("clockwork periods must be positive")
            self.periods = ps
            self.blocks = len(ps)
            # split hidden evenly
            sizes = [hid // self.blocks] * self.blocks
            for i in range(hid - sum(sizes)):
                sizes[i] += 1
            self.sizes = tuple(sizes)
            self.x2h = nn.ModuleList([nn.Linear(1, s, bias=True) for s in self.sizes])
            self.h2h = nn.ModuleList([nn.Linear(s, s, bias=False) for s in self.sizes])

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            hs = [
                torch.zeros((int(B), int(s)), device=xb.device, dtype=xb.dtype) for s in self.sizes
            ]
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                for i, p in enumerate(self.periods):
                    if (t % int(p)) == 0:
                        hs[i] = torch.tanh(self.x2h[i](x_t) + self.h2h[i](hs[i]))
                outs.append(torch.cat(hs, dim=-1).unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _QRNNEncoder(_SeqEncoder):
        def __init__(self, *, k: int) -> None:
            super().__init__()
            kk = int(k)
            if kk <= 0:
                raise ValueError("kernel_size must be >= 1")
            self.k = kk
            self.conv = nn.Conv1d(1, 3 * hid, kernel_size=int(kk), padding=int(kk) - 1)

        def forward(self, xb: Any) -> Any:
            # xb: (B,T,1) -> conv on time axis
            x1 = xb.transpose(1, 2)  # (B,1,T)
            gates = self.conv(x1)[:, :, : xb.shape[1]]  # causal-ish crop
            z, f_gate, o_gate = gates.chunk(3, dim=1)  # (B,H,T)
            z = torch.tanh(z).transpose(1, 2)
            f_gate = torch.sigmoid(f_gate).transpose(1, 2)
            o_gate = torch.sigmoid(o_gate).transpose(1, 2)

            B, T, _ = z.shape
            c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                c_t = f_gate[:, t, :] * c_t + (1.0 - f_gate[:, t, :]) * z[:, t, :]
                h_t = o_gate[:, t, :] * c_t
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _PhasedLSTMEncoder(_SeqEncoder):
        def __init__(self, *, tau: float, r_on: float, leak: float) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, 4 * hid, bias=True)
            self.h2h = nn.Linear(hid, 4 * hid, bias=False)
            self.tau = float(tau)
            self.r_on = float(r_on)
            self.leak = float(leak)
            # phase offset
            self.phase = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            tau = float(self.tau)
            r_on = float(self.r_on)
            leak = float(self.leak)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                gates = self.x2h(x_t) + self.h2h(h_t)
                i, f, g, o = gates.chunk(4, dim=-1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c_new = f * c_t + i * g
                h_new = o * torch.tanh(c_new)

                # time gate in [0,1]
                tt = torch.tensor(float(t), device=xb.device, dtype=xb.dtype)
                phi = torch.remainder((tt + self.phase) / max(1e-6, tau), 1.0)
                k_t = torch.where(
                    phi < r_on,
                    phi / max(1e-6, r_on),
                    torch.where(
                        phi < (2.0 * r_on),
                        (2.0 * r_on - phi) / max(1e-6, r_on),
                        leak * phi,
                    ),
                )
                k_t = torch.clamp(k_t, 0.0, 1.0)

                c_t = k_t * c_new + (1.0 - k_t) * c_t
                h_t = k_t * h_new + (1.0 - k_t) * h_t
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _ESNEncoder(_SeqEncoder):
        def __init__(self, *, spectral_radius: float, leak: float) -> None:
            super().__init__()
            sr = float(spectral_radius)
            if sr <= 0.0:
                raise ValueError("spectral_radius must be > 0")
            lk = float(leak)
            if not (0.0 < lk <= 1.0):
                raise ValueError("leak must be in (0,1]")
            self.leak = lk

            # fixed random reservoir weights
            rng = np.random.default_rng(int(seed))
            w_in = rng.standard_normal((1, hid)).astype(np.float32) / max(1.0, float(hid) ** 0.5)
            w = rng.standard_normal((hid, hid)).astype(np.float32) / max(1.0, float(hid) ** 0.5)
            # scale to target spectral radius (approx via power iteration)
            v = rng.standard_normal((hid,)).astype(np.float32)
            for _ in range(20):
                v = w @ v
                nrm = float(np.linalg.norm(v))
                if nrm > 0:
                    v /= nrm
            eig_est = float(np.linalg.norm(w @ v) / max(1e-6, np.linalg.norm(v)))
            scale = sr / max(1e-6, eig_est)
            w *= float(scale)

            self.register_buffer("w_in", torch.tensor(w_in, dtype=torch.float32))
            self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            lk = float(self.leak)
            for t in range(int(T)):
                x_t = xb[:, t, :]  # (B,1)
                pre = x_t @ self.w_in + h_t @ self.w
                h_new = torch.tanh(pre)
                h_t = (1.0 - lk) * h_t + lk * h_new
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _DeepESNEncoder(_SeqEncoder):
        """
        Deep Echo State Network (lite): stack multiple fixed reservoirs.

        Each reservoir layer is untrained and maps its input sequence to a hidden
        state sequence; the next reservoir consumes the previous layer's states.
        """

        class _ReservoirLayer(nn.Module):
            def __init__(
                self, *, in_dim: int, spectral_radius: float, leak: float, seed: int
            ) -> None:
                super().__init__()
                sr = float(spectral_radius)
                if sr <= 0.0:
                    raise ValueError("spectral_radius must be > 0")
                lk = float(leak)
                if not (0.0 < lk <= 1.0):
                    raise ValueError("leak must be in (0,1]")
                self.leak = lk

                rng = np.random.default_rng(int(seed))
                w_in = rng.standard_normal((int(in_dim), hid)).astype(np.float32) / max(
                    1.0, float(hid) ** 0.5
                )
                w = rng.standard_normal((hid, hid)).astype(np.float32) / max(1.0, float(hid) ** 0.5)

                v = rng.standard_normal((hid,)).astype(np.float32)
                for _ in range(20):
                    v = w @ v
                    nrm = float(np.linalg.norm(v))
                    if nrm > 0:
                        v /= nrm
                eig_est = float(np.linalg.norm(w @ v) / max(1e-6, np.linalg.norm(v)))
                w *= float(sr / max(1e-6, eig_est))

                self.register_buffer("w_in", torch.tensor(w_in, dtype=torch.float32))
                self.register_buffer("w", torch.tensor(w, dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                lk = float(self.leak)
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    pre = x_t @ self.w_in + h_t @ self.w
                    h_new = torch.tanh(pre)
                    h_t = (1.0 - lk) * h_t + lk * h_new
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        def __init__(self, *, depth: int, spectral_radius: float, leak: float) -> None:
            super().__init__()
            d = int(depth)
            if d <= 0:
                raise ValueError("depth must be >= 1")
            self.layers = nn.ModuleList(
                [
                    self._ReservoirLayer(
                        in_dim=(1 if i == 0 else hid),
                        spectral_radius=float(spectral_radius),
                        leak=float(leak),
                        seed=int(seed) + 13 * i,
                    )
                    for i in range(d)
                ]
            )

        def forward(self, xb: Any) -> Any:
            out = xb
            for layer in self.layers:
                out = layer(out)
            return out

    class _ConceptorESNEncoder(_SeqEncoder):
        def __init__(self, *, spectral_radius: float, leak: float) -> None:
            super().__init__()
            self.esn = _ESNEncoder(spectral_radius=spectral_radius, leak=leak)
            self.C = nn.Parameter(torch.eye(hid, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            seq = self.esn(xb)  # (B,T,H)
            # apply conceptor as linear projection per time step
            return torch.einsum("bth,hk->btk", seq, self.C)

    class _DotAttentionPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q = nn.Linear(hid, hid, bias=True)

        def forward(self, seq: Any) -> Any:
            # seq: (B,T,H), query = last state
            q = self.q(seq[:, -1, :]).unsqueeze(2)  # (B,H,1)
            scores = torch.bmm(seq, q).squeeze(2) / max(1.0, float(hid) ** 0.5)  # (B,T)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
            return torch.sum(w * seq, dim=1)  # (B,H)

    class _AdditiveAttentionPool(nn.Module):
        def __init__(self, *, attn_hidden: int) -> None:
            super().__init__()
            a = int(attn_hidden)
            if a <= 0:
                raise ValueError("attn_hidden must be >= 1")
            self.W = nn.Linear(hid, a, bias=True)
            self.v = nn.Linear(a, 1, bias=False)

        def forward(self, seq: Any) -> Any:
            scores = self.v(torch.tanh(self.W(seq))).squeeze(-1)  # (B,T)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            return torch.sum(w * seq, dim=1)

    class _RWAEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2a = nn.Linear(1, hid, bias=True)
            self.h2a = nn.Linear(hid, hid, bias=False)
            self.x2u = nn.Linear(1, hid, bias=True)
            self.h2u = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            n_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            d_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            h_prev = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                a_t = torch.tanh(self.x2a(x_t) + self.h2a(h_prev))
                u_t = self.x2u(x_t) + self.h2u(h_prev)
                g_t = torch.exp(torch.clamp(u_t, -20.0, 20.0))
                n_t = n_t + g_t * a_t
                d_t = d_t + g_t
                h_prev = n_t / (d_t + 1e-6)
                outs.append(h_prev.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _AntisymmetricRNNEncoder(_SeqEncoder):
        def __init__(self, *, gamma: float = 0.01, dt: float = 0.1) -> None:
            super().__init__()
            self.dt = float(dt)
            self.gamma = float(gamma)
            self.V = nn.Linear(1, hid, bias=True)
            self.A = nn.Parameter(torch.zeros((hid, hid), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            dt = float(self.dt)
            gamma = float(self.gamma)
            eye = torch.eye(hid, device=xb.device, dtype=xb.dtype)
            W = self.A - self.A.transpose(0, 1) - gamma * eye
            for t in range(int(T)):
                x_t = xb[:, t, :]
                dh = torch.tanh(self.V(x_t) + h_t @ W)
                h_t = h_t + dt * dh
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _CoRNNEncoder(_SeqEncoder):
        def __init__(self, *, dt: float = 0.1, gamma: float = 0.1, omega: float = 1.0) -> None:
            super().__init__()
            self.dt = float(dt)
            self.gamma = float(gamma)
            self.omega = float(omega)
            self.Wx = nn.Linear(1, hid, bias=True)
            self.Wh = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            y = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            z = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            dt = float(self.dt)
            gamma = float(self.gamma)
            omega2 = float(self.omega) ** 2
            for t in range(int(T)):
                x_t = xb[:, t, :]
                z = z + dt * (torch.tanh(self.Wx(x_t) + self.Wh(y)) - gamma * z - omega2 * y)
                y = y + dt * z
                outs.append(y.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _UnICORNNEncoder(_SeqEncoder):
        """
        UnICORNN-inspired stable recurrence (lite):
        rotate a 2nd-order state (y,z) and inject input forcing with learnable damping.
        """

        def __init__(self, *, dt: float = 0.1) -> None:
            super().__init__()
            self.dt = float(dt)
            self.Wx = nn.Linear(1, hid, bias=True)
            self.Wh = nn.Linear(hid, hid, bias=False)
            self.omega_log = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
            self.gamma_logit = nn.Parameter(torch.full((hid,), -2.0, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            y = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            z = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            dt = float(self.dt)
            omega = torch.exp(self.omega_log).to(device=xb.device, dtype=xb.dtype)  # (H,)
            gamma = torch.sigmoid(self.gamma_logit).to(device=xb.device, dtype=xb.dtype)  # (H,)
            theta = dt * omega
            c = torch.cos(theta).unsqueeze(0)  # (1,H)
            s = torch.sin(theta).unsqueeze(0)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                y_rot = c * y + s * z
                z_rot = -s * y + c * z
                force = torch.tanh(self.Wx(x_t) + self.Wh(y_rot))
                z = (1.0 - gamma) * z_rot + dt * force
                y = y_rot + dt * z
                outs.append(y.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _LEMEncoder(_SeqEncoder):
        """
        Long Expressive Memory (lite): fast gated state + slow exponentially-smoothed memory.
        """

        def __init__(self) -> None:
            super().__init__()
            self.x2u = nn.Linear(1, hid, bias=True)
            self.h2u = nn.Linear(hid, hid, bias=False)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)
            self.alpha_logit = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            m_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            alpha = torch.sigmoid(self.alpha_logit).to(device=xb.device, dtype=xb.dtype)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                u = torch.sigmoid(self.x2u(x_t) + self.h2u(h_t))
                cand = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                h_t = (1.0 - u) * h_t + u * cand
                m_t = (1.0 - alpha) * m_t + alpha * h_t
                outs.append(torch.tanh(h_t + m_t).unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _ResidualRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.h2h = nn.Linear(hid, hid, bias=False)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_t = h_t + torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _OrthogonalRNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.W = nn.Parameter(torch.empty((hid, hid), dtype=torch.float32))
            nn.init.orthogonal_(self.W)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            # project to orthogonal each forward (cheap for small hid)
            Q, _R = torch.linalg.qr(self.W)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_t = torch.tanh(self.x2h(x_t) + h_t @ Q)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _UnitaryCayleyRNNEncoder(_SeqEncoder):
        """
        Unitary/orthogonal transition via Cayley transform of a skew-symmetric matrix.

        This avoids QR re-projection and yields an exactly orthogonal matrix in exact arithmetic:
        Q = (I + S)^{-1} (I - S) where S = A - A^T.
        """

        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.A = nn.Parameter(torch.zeros((hid, hid), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            eye = torch.eye(hid, device=xb.device, dtype=xb.dtype)
            S = self.A - self.A.transpose(0, 1)
            Q = torch.linalg.solve(eye + S, eye - S)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_t = torch.tanh(self.x2h(x_t) + h_t @ Q)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _EUNNEncoder(_SeqEncoder):
        """
        Efficient Unitary Neural Network (lite): parameterize an orthogonal transform
        as a small number of 2脳2 Givens rotations (even/odd pairing).
        """

        def __init__(self) -> None:
            super().__init__()
            self.x2h = nn.Linear(1, hid, bias=True)
            self.theta_even = nn.Parameter(torch.zeros((hid // 2,), dtype=torch.float32))
            self.theta_odd = nn.Parameter(torch.zeros(((hid - 1) // 2,), dtype=torch.float32))

        @staticmethod
        def _apply_rot(h: Any, theta: Any, *, offset: int) -> Any:
            if int(offset) not in {0, 1}:
                raise ValueError("offset must be 0 or 1")
            if int(offset) == 0:
                a = h[:, 0::2]
                b = h[:, 1::2]
            else:
                a = h[:, 1::2]
                b = h[:, 2::2]
            pairs = min(int(a.shape[1]), int(b.shape[1]), int(theta.shape[0]))
            if pairs <= 0:
                return h
            c = torch.cos(theta[:pairs]).unsqueeze(0).to(dtype=h.dtype, device=h.device)
            s = torch.sin(theta[:pairs]).unsqueeze(0).to(dtype=h.dtype, device=h.device)
            a2 = a[:, :pairs]
            b2 = b[:, :pairs]
            new_a = c * a2 - s * b2
            new_b = s * a2 + c * b2
            if int(offset) == 0:
                h = h.clone()
                h[:, 0 : 2 * pairs : 2] = new_a
                h[:, 1 : 2 * pairs : 2] = new_b
            else:
                h = h.clone()
                h[:, 1 : 1 + 2 * pairs : 2] = new_a
                h[:, 2 : 2 + 2 * pairs : 2] = new_b
            return h

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_rot = self._apply_rot(h_t, self.theta_even, offset=0)
                h_rot = self._apply_rot(h_rot, self.theta_odd, offset=1)
                h_t = torch.tanh(self.x2h(x_t) + h_rot)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _GORUEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.x2z = nn.Linear(1, hid, bias=True)
            self.x2r = nn.Linear(1, hid, bias=True)
            self.x2h = nn.Linear(1, hid, bias=True)
            self.W = nn.Parameter(torch.empty((hid, hid), dtype=torch.float32))
            nn.init.orthogonal_(self.W)
            self.Uz = nn.Parameter(torch.zeros((hid, hid), dtype=torch.float32))
            self.Ur = nn.Parameter(torch.zeros((hid, hid), dtype=torch.float32))

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            Q, _R = torch.linalg.qr(self.W)
            for t in range(int(T)):
                x_t = xb[:, t, :]
                z = torch.sigmoid(self.x2z(x_t) + h_t @ self.Uz)
                r = torch.sigmoid(self.x2r(x_t) + h_t @ self.Ur)
                h_hat = torch.tanh(self.x2h(x_t) + (r * h_t) @ Q)
                h_t = (1.0 - z) * h_t + z * h_hat
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _ODERNNEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.f = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, hid))
            self.gru = _GRUCell(in_dim=1)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                # ODE drift step
                h_t = h_t + 0.1 * self.f(h_t)
                # Observation update
                x_t = xb[:, t, :]
                h_t, _ = self.gru.step(x_t, h_t, t=t)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _NeuralCDEEncoder(_SeqEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.f = nn.Sequential(nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, hid))
            self.x0 = nn.Linear(1, hid, bias=True)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            h_t = self.x0(xb[:, 0, :])
            outs = [h_t.unsqueeze(1)]
            for t in range(1, int(T)):
                dx = xb[:, t, :] - xb[:, t - 1, :]
                h_t = h_t + 0.1 * self.f(h_t) * dx
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _MemoryNetEncoder(_SeqEncoder):
        def __init__(self, *, hops: int, attn_hidden: int) -> None:
            super().__init__()
            hp = int(hops)
            if hp <= 0:
                raise ValueError("hops must be >= 1")
            self.hops = hp
            self.emb = nn.Linear(1, hid, bias=True)
            self.q_proj = nn.Linear(hid, hid, bias=True)
            self.k_proj = nn.Linear(hid, hid, bias=True)
            self.v_proj = nn.Linear(hid, hid, bias=True)
            self.ff = nn.Sequential(
                nn.Linear(hid, int(attn_hidden)),
                nn.Tanh(),
                nn.Linear(int(attn_hidden), hid),
            )

        def forward(self, xb: Any) -> Any:
            mem = self.emb(xb)  # (B,T,H)
            q = mem[:, -1, :]  # (B,H)
            for _ in range(self.hops):
                qk = self.q_proj(q).unsqueeze(2)  # (B,H,1)
                keys = self.k_proj(mem)
                vals = self.v_proj(mem)
                scores = torch.bmm(keys, qk).squeeze(2) / max(1.0, float(hid) ** 0.5)
                w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
                ctx = torch.sum(w * vals, dim=1)
                q = q + self.ff(ctx)
            # return a "sequence" by repeating query; downstream takes last anyway
            return q.unsqueeze(1).expand(mem.shape[0], mem.shape[1], q.shape[1])

    class _NTMEncoder(_SeqEncoder):
        def __init__(self, *, slots: int, mem_dim: int) -> None:
            super().__init__()
            m = int(slots)
            d = int(mem_dim)
            if m <= 0:
                raise ValueError("memory_slots must be >= 1")
            if d <= 0:
                raise ValueError("memory_dim must be >= 1")
            self.M = m
            self.D = d
            self.controller = _GRUCell(in_dim=1 + d)
            self.key = nn.Linear(hid, d, bias=True)
            self.add = nn.Linear(hid, d, bias=True)
            self.erase = nn.Linear(hid, d, bias=True)
            self.beta = nn.Linear(hid, 1, bias=True)

        def forward(self, xb: Any) -> Any:
            B, T, _ = xb.shape
            mem = torch.zeros((int(B), self.M, self.D), device=xb.device, dtype=xb.dtype)
            w = torch.zeros((int(B), self.M), device=xb.device, dtype=xb.dtype)
            h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
            r_t = torch.zeros((int(B), self.D), device=xb.device, dtype=xb.dtype)
            outs = []
            for t in range(int(T)):
                x_t = xb[:, t, :]
                h_t, _ = self.controller.step(torch.cat([x_t, r_t], dim=-1), h_t, t=t)
                k = self.key(h_t)  # (B,D)
                b = F.softplus(self.beta(h_t)) + 1e-3
                # cosine similarity
                mem_norm = mem / (torch.norm(mem, dim=-1, keepdim=True) + 1e-6)
                k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
                sim = torch.sum(mem_norm * k_norm.unsqueeze(1), dim=-1)  # (B,M)
                w = torch.softmax(b * sim, dim=1)

                e = torch.sigmoid(self.erase(h_t)).unsqueeze(1)  # (B,1,D)
                a = torch.tanh(self.add(h_t)).unsqueeze(1)
                mem = mem * (1.0 - w.unsqueeze(-1) * e) + w.unsqueeze(-1) * a
                r_t = torch.sum(w.unsqueeze(-1) * mem, dim=1)
                outs.append(h_t.unsqueeze(1))
            return torch.cat(outs, dim=1)

    class _RNNPaperNet(nn.Module):
        def __init__(self, *, encoder: _SeqEncoder, pool: str = "last") -> None:
            super().__init__()
            self.encoder = encoder
            self.pool = str(pool).lower().strip()
            if self.pool == "dot":
                self.pooler = _DotAttentionPool()
            elif self.pool == "additive":
                self.pooler = _AdditiveAttentionPool(attn_hidden=int(attn_hidden))
            else:
                self.pooler = None
            self.head = nn.Linear(hid, h)

        def forward(self, xb: Any) -> Any:
            out_seq = self.encoder(xb)
            if self.pooler is None:
                ctx = out_seq[:, -1, :]
            else:
                ctx = self.pooler(out_seq)
            return self.head(ctx)

    # ---- Encoder selection ----
    # Many paper ids share core building blocks; we keep them distinct by key/description.
    if paper_id in {"elman-srn"}:
        encoder: _SeqEncoder = _ScanEncoder(_RNNCell())
    elif paper_id in {"jordan-srn"}:
        encoder = _ScanEncoder(_JordanCell())
    elif paper_id in {"bidirectional-rnn"}:
        rnn_drop = float(drop) if int(layers) > 1 else 0.0
        fwd = _StackedScanEncoder(
            [_RNNCell(in_dim=(1 if i == 0 else hid)) for i in range(int(layers))],
            dropout=rnn_drop,
        )
        bwd = _StackedScanEncoder(
            [_RNNCell(in_dim=(1 if i == 0 else hid)) for i in range(int(layers))],
            dropout=rnn_drop,
        )

        class _Bidirectional(_SeqEncoder):
            def __init__(self, *, fwd: Any, bwd: Any) -> None:
                super().__init__()
                self.fwd = fwd
                self.bwd = bwd
                self.proj = nn.Linear(2 * hid, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                out_f = self.fwd(xb)
                out_b = self.bwd(torch.flip(xb, dims=[1]))
                out_b = torch.flip(out_b, dims=[1])
                return self.proj(torch.cat([out_f, out_b], dim=-1))

        encoder = _Bidirectional(fwd=fwd, bwd=bwd)
    elif paper_id in {"multi-dimensional-rnn"}:
        # MDRNN-lite: reshape the lag window into a 2D grid and scan left+up dependencies.
        class _MDRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.wx = nn.Linear(1, hid, bias=True)
                self.wh_l = nn.Linear(hid, hid, bias=False)
                self.wh_u = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                side = int(max(1, round(float(T) ** 0.5)))
                Hh = side
                Ww = int((int(T) + Hh - 1) // Hh)
                pad = Hh * Ww - int(T)
                if pad > 0:
                    xb2 = torch.cat(
                        [xb, torch.zeros((int(B), pad, 1), device=xb.device, dtype=xb.dtype)], dim=1
                    )
                else:
                    xb2 = xb
                grid = xb2.reshape(int(B), Hh, Ww, 1)
                zero_state = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                rows: list[Any] = []
                for i in range(Hh):
                    row_states = []
                    for j in range(Ww):
                        x_ij = grid[:, i, j, :]
                        h_l = row_states[j - 1] if j > 0 else zero_state
                        h_u = rows[i - 1][j] if i > 0 else zero_state
                        h_ij = torch.tanh(self.wx(x_ij) + self.wh_l(h_l) + self.wh_u(h_u))
                        row_states.append(h_ij)
                    rows.append(row_states)
                h_grid = torch.stack([torch.stack(row, dim=1) for row in rows], dim=1)
                last = h_grid[:, -1, -1, :].unsqueeze(1)
                return last.expand(int(B), int(T), hid)

        encoder = _MDRNN()
    elif paper_id in {"gated-feedback-rnn"}:

        class _GFRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2h1 = nn.Linear(1, hid, bias=True)
                self.h12h1 = nn.Linear(hid, hid, bias=False)
                self.h22h1 = nn.Linear(hid, hid, bias=False)
                self.gate = nn.Linear(hid, hid, bias=True)
                self.h12h2 = nn.Linear(hid, hid, bias=True)
                self.h22h2 = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h1 = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                h2 = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    g = torch.sigmoid(self.gate(h2))
                    h1 = torch.tanh(self.x2h1(x_t) + self.h12h1(h1) + self.h22h1(g * h2))
                    h2 = torch.tanh(self.h12h2(h1) + self.h22h2(h2))
                    outs.append(h2.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _GFRNN()
    elif paper_id in {"hierarchical-multiscale-rnn"}:

        class _HMRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.cell1 = _RNNCell()
                self.cell2 = _RNNCell(in_dim=hid)
                self.boundary = nn.Linear(hid, 1, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h1 = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                h2 = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h1, h1_out = self.cell1.step(x_t, h1, t=t)
                    b = torch.sigmoid(self.boundary(h1_out))  # (B,1)
                    h2_new, h2_out = self.cell2.step(h1_out, h2, t=t)
                    h2 = b * h2_new + (1.0 - b) * h2
                    outs.append(h2_out.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _HMRNN()
    elif paper_id in {"clockwork-rnn"}:
        encoder = _ClockworkEncoder()
    elif paper_id in {"dilated-rnn"}:

        class _Dilated(_SeqEncoder):
            def __init__(self, *, dilation: int = 2) -> None:
                super().__init__()
                self.d = int(dilation)
                if self.d <= 0:
                    raise ValueError("dilation must be >= 1")
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                buf = [
                    torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                    for _ in range(self.d + 1)
                ]
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_d = buf[t % (self.d + 1)]
                    h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_d))
                    buf[t % (self.d + 1)] = h_t
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _Dilated(dilation=2)
    elif paper_id in {"skip-rnn"}:

        class _Skip(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.core = _RNNCell()
                self.u = nn.Linear(1 + hid, 1, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_new, _h_out = self.core.step(x_t, h_t, t=t)
                    u_t = torch.sigmoid(self.u(torch.cat([x_t, h_t], dim=-1)))
                    h_t = u_t * h_new + (1.0 - u_t) * h_t
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _Skip()
    elif paper_id in {"sliced-rnn"}:

        class _Sliced(_SeqEncoder):
            def __init__(self, *, slice_len: int = 4) -> None:
                super().__init__()
                self.s = int(slice_len)
                if self.s <= 0:
                    raise ValueError("slice_len must be >= 1")
                self.inner = _RNNCell()
                self.outer = _RNNCell(in_dim=hid)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                # inner over slices
                slice_states = []
                for start in range(0, int(T), self.s):
                    h_in = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                    for t in range(start, min(int(T), start + self.s)):
                        h_in, _ = self.inner.step(xb[:, t, :], h_in, t=t)
                    slice_states.append(h_in)
                # outer over slice summaries
                h_out = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for i, s in enumerate(slice_states):
                    h_out, _ = self.outer.step(s, h_out, t=i)
                    outs.append(h_out)
                last = outs[-1] if outs else h_out
                return last.unsqueeze(1).expand(int(B), int(T), hid)

        encoder = _Sliced(slice_len=4)
    elif paper_id in {"lstm", "forget-gate-lstm", "chrono-lstm"}:
        rnn_drop = float(drop) if int(layers) > 1 else 0.0
        forget_bias_init = None
        input_bias_init = None
        if paper_id == "forget-gate-lstm":
            forget_bias_init = 1.0
        elif paper_id == "chrono-lstm":
            tau = max(2.0, float(lag_count))
            b = float(np.log(tau - 1.0))
            forget_bias_init = b
            input_bias_init = -b

        encoder = _StackedScanEncoder(
            [
                _LSTMCell(
                    in_dim=(1 if i == 0 else hid),
                    forget_bias_init=forget_bias_init,
                    input_bias_init=input_bias_init,
                )
                for i in range(int(layers))
            ],
            dropout=rnn_drop,
        )
    elif paper_id in {"peephole-lstm"}:
        encoder = _ScanEncoder(_PeepholeLSTMCell())
    elif paper_id in {"lstm-projection"}:
        proj_size = max(1, hid // 2)
        rnn_drop = float(drop) if int(layers) > 1 else 0.0

        core = _StackedScanEncoder(
            [
                _LSTMPCell(in_dim=(1 if i == 0 else int(proj_size)), proj_dim=int(proj_size))
                for i in range(int(layers))
            ],
            dropout=rnn_drop,
        )

        class _LSTMPWrap(_SeqEncoder):
            def __init__(self, core: Any, *, proj: int) -> None:
                super().__init__()
                self.core = core
                self.proj = int(proj)
                self.to_hid = nn.Linear(self.proj, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                out = self.core(xb)  # (B,T,proj)
                return self.to_hid(out)

        encoder = _LSTMPWrap(core, proj=int(proj_size))
    elif paper_id in {"cifg-lstm"}:
        encoder = _ScanEncoder(_CIFGLSTMCell())
    elif paper_id in {"phased-lstm"}:
        encoder = _PhasedLSTMEncoder(tau=max(2.0, float(lag_count)), r_on=0.05, leak=0.001)
    elif paper_id in {"grid-lstm"}:
        # Grid LSTM-lite: treat "grid" as stacked LSTM.
        grid_layers = max(2, int(layers))
        rnn_drop = float(drop) if int(grid_layers) > 1 else 0.0
        encoder = _StackedScanEncoder(
            [_LSTMCell(in_dim=(1 if i == 0 else hid)) for i in range(int(grid_layers))],
            dropout=rnn_drop,
        )
    elif paper_id in {"tree-lstm"}:

        class _TreeLSTM(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.leaf = nn.Linear(1, 2 * hid, bias=True)
                self.h2i_l = nn.Linear(hid, hid, bias=False)
                self.h2i_r = nn.Linear(hid, hid, bias=False)
                self.h2f_l = nn.Linear(hid, hid, bias=False)
                self.h2f_r = nn.Linear(hid, hid, bias=False)
                self.h2o_l = nn.Linear(hid, hid, bias=False)
                self.h2o_r = nn.Linear(hid, hid, bias=False)
                self.h2u_l = nn.Linear(hid, hid, bias=False)
                self.h2u_r = nn.Linear(hid, hid, bias=False)

            def _combine(self, left: tuple[Any, Any], right: tuple[Any, Any]) -> tuple[Any, Any]:
                h_l, c_l = left
                h_r, c_r = right
                i = torch.sigmoid(self.h2i_l(h_l) + self.h2i_r(h_r))
                f_l = torch.sigmoid(self.h2f_l(h_l))
                f_r = torch.sigmoid(self.h2f_r(h_r))
                o = torch.sigmoid(self.h2o_l(h_l) + self.h2o_r(h_r))
                u = torch.tanh(self.h2u_l(h_l) + self.h2u_r(h_r))
                c = i * u + f_l * c_l + f_r * c_r
                h = o * torch.tanh(c)
                return (h, c)

            def forward(self, xb: Any) -> Any:
                # leaves
                leaves: list[tuple[Any, Any]] = []
                for t in range(int(xb.shape[1])):
                    z = self.leaf(xb[:, t, :])
                    h0, c0 = z.chunk(2, dim=-1)
                    leaves.append((torch.tanh(h0), torch.tanh(c0)))
                nodes = leaves
                while len(nodes) > 1:
                    nxt: list[tuple[Any, Any]] = []
                    for i in range(0, len(nodes), 2):
                        if i + 1 < len(nodes):
                            nxt.append(self._combine(nodes[i], nodes[i + 1]))
                        else:
                            nxt.append(nodes[i])
                    nodes = nxt
                h_root, _c_root = nodes[0]
                return h_root.unsqueeze(1).expand(xb.shape[0], xb.shape[1], h_root.shape[1])

        encoder = _TreeLSTM()
    elif paper_id in {"nested-lstm"}:

        class _NestedLSTM(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _LSTMCell(in_dim=1)
                self.outer_x = nn.Linear(1, 4 * hid, bias=True)
                self.outer_h = nn.Linear(hid, 4 * hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_o = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_o = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                h_i = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_i = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    # inner processes x to produce candidate
                    (h_i, c_i), _h_i = self.inner.step(x_t, (h_i, c_i), t=t)
                    gates = self.outer_x(x_t) + self.outer_h(h_o)
                    i, f, g, o = gates.chunk(4, dim=-1)
                    i = torch.sigmoid(i)
                    f = torch.sigmoid(f)
                    g = torch.tanh(g) + 0.1 * h_i
                    o = torch.sigmoid(o)
                    c_o = f * c_o + i * g
                    h_o = o * torch.tanh(c_o)
                    outs.append(h_o.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _NestedLSTM()
    elif paper_id in {"on-lstm"}:

        def _cumax(z: Any) -> Any:
            return torch.cumsum(torch.softmax(z, dim=-1), dim=-1)

        class _ONLSTM(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2h = nn.Linear(1, 4 * hid, bias=True)
                self.h2h = nn.Linear(hid, 4 * hid, bias=False)
                self.x2mf = nn.Linear(1, hid, bias=True)
                self.h2mf = nn.Linear(hid, hid, bias=False)
                self.x2mi = nn.Linear(1, hid, bias=True)
                self.h2mi = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    gates = self.x2h(x_t) + self.h2h(h_t)
                    i, f, g, o = gates.chunk(4, dim=-1)
                    i = torch.sigmoid(i)
                    f = torch.sigmoid(f)
                    g = torch.tanh(g)
                    o = torch.sigmoid(o)
                    mf = _cumax(self.x2mf(x_t) + self.h2mf(h_t))
                    mi = 1.0 - _cumax(self.x2mi(x_t) + self.h2mi(h_t))
                    c_t = mf * (f * c_t) + mi * (i * g)
                    h_t = o * torch.tanh(c_t)
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _ONLSTM()
    elif paper_id in {"lattice-rnn"}:

        class _LatticeRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.cell = _RNNCell()
                self.mix = nn.Linear(hid * 2, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                hs = []
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_t, _ = self.cell.step(x_t, h_t, t=t)
                    hs.append(h_t)
                outs = []
                for t in range(int(T)):
                    h_prev = hs[t - 1] if t > 0 else torch.zeros_like(hs[0])
                    h_skip = hs[t - 2] if t > 1 else torch.zeros_like(hs[0])
                    outs.append(
                        torch.tanh(self.mix(torch.cat([h_prev, h_skip], dim=-1))).unsqueeze(1)
                    )
                return torch.cat(outs, dim=1)

        encoder = _LatticeRNN()
    elif paper_id in {"lattice-lstm", "wmc-lstm"}:

        class _LatticeLSTM(_SeqEncoder):
            def __init__(self, *, use_wmc: bool) -> None:
                super().__init__()
                self.use_wmc = bool(use_wmc)
                self.x2h = nn.Linear(1, 4 * hid, bias=True)
                self.h2h = nn.Linear(hid, 4 * hid, bias=False)
                self.c_proj = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_tm2 = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    gates = self.x2h(x_t) + self.h2h(h_t)
                    if self.use_wmc:
                        gates = gates + self.c_proj(torch.tanh(c_t)).repeat(1, 4)
                    i, f, g, o = gates.chunk(4, dim=-1)
                    i = torch.sigmoid(i)
                    f = torch.sigmoid(f)
                    g = torch.tanh(g)
                    o = torch.sigmoid(o)
                    # lattice: mix c_{t-1} and c_{t-2}
                    mix = 0.5
                    c_prev = mix * c_t + (1.0 - mix) * c_tm2
                    c_tm2 = c_t
                    c_t = f * c_prev + i * g
                    h_t = o * torch.tanh(c_t)
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _LatticeLSTM(use_wmc=(paper_id == "wmc-lstm"))
    elif paper_id in {"gru", "gru-variant-1", "gru-variant-2", "gru-variant-3"}:
        variant = 0
        if paper_id == "gru-variant-1":
            variant = 1
        elif paper_id == "gru-variant-2":
            variant = 2
        elif paper_id == "gru-variant-3":
            variant = 3
        encoder = _ScanEncoder(_GRUVariantCell(variant=int(variant)))
    elif paper_id in {"mgu", "mgu1", "mgu2", "mgu3"}:
        mode = "full"
        if paper_id == "mgu1":
            mode = "h-only"
        elif paper_id == "mgu2":
            mode = "x-only"
        elif paper_id == "mgu3":
            mode = "bias-only"
        encoder = _ScanEncoder(_MGUCell(gate_mode=mode))
    elif paper_id in {"ligru"}:

        class _LiGRU(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2z = nn.Linear(1, hid, bias=True)
                self.h2z = nn.Linear(hid, hid, bias=False)
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)
                self.ln = nn.LayerNorm(hid)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_t))
                    h_hat = torch.relu(self.ln(self.x2h(x_t) + self.h2h(h_t)))
                    h_t = (1.0 - z) * h_t + z * h_hat
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _LiGRU()
    elif paper_id in {"sru"}:
        encoder = _ScanEncoder(_SRUCell())
    elif paper_id in {"qrnn"}:
        encoder = _QRNNEncoder(k=int(kernel_size))
    elif paper_id in {"indrnn"}:
        encoder = _ScanEncoder(_IndRNNCell())
    elif paper_id in {"minimalrnn"}:
        encoder = _ScanEncoder(_MinimalRNNCell())
    elif paper_id in {"cfn"}:

        class _CFN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)
                self.alpha_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                alpha = torch.sigmoid(self.alpha_logit)
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_new = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                    h_t = alpha * h_t + (1.0 - alpha) * h_new
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _CFN()
    elif paper_id in {"ran"}:
        encoder = _ScanEncoder(_RANCell())
    elif paper_id in {"atr"}:

        class _ATR(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2g1 = nn.Linear(1, hid, bias=True)
                self.h2g1 = nn.Linear(hid, hid, bias=False)
                self.x2g2 = nn.Linear(1, hid, bias=True)
                self.h2g2 = nn.Linear(hid, hid, bias=False)
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    g_add = torch.sigmoid(self.x2g1(x_t) + self.h2g1(h_t))
                    g_sub = torch.sigmoid(self.x2g2(x_t) - self.h2g2(h_t))
                    h_hat = torch.tanh(self.x2h(x_t) + self.h2h(g_add * h_t))
                    h_t = g_sub * h_t + (1.0 - g_sub) * h_hat
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _ATR()
    elif paper_id in {"mut1", "mut2", "mut3"}:
        v = {"mut1": 1, "mut2": 2, "mut3": 3}[paper_id]
        encoder = _ScanEncoder(_MUTCell(variant=int(v)))
    elif paper_id in {"fast-rnn"}:
        encoder = _ScanEncoder(_FastRNNCell())
    elif paper_id in {"fast-grnn"}:
        encoder = _ScanEncoder(_FastGRNNCell())
    elif paper_id in {"fru"}:

        class _FRU(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)
                self.freq = nn.Parameter(torch.linspace(0.0, 3.14, hid))
                self.g = nn.Linear(1, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                m_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_t = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                    tt = torch.tensor(float(t), device=xb.device, dtype=xb.dtype)
                    basis = torch.cos(tt * self.freq)
                    gate = torch.sigmoid(self.g(x_t))
                    m_t = (1.0 - gate) * m_t + gate * (h_t * basis)
                    outs.append(torch.tanh(m_t).unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _FRU()
    elif paper_id in {"rwa"}:
        encoder = _RWAEncoder()
    elif paper_id in {"rhn"}:
        encoder = _ScanEncoder(_RHNCell(depth=2))
    elif paper_id in {"scrn"}:
        encoder = _ScanEncoder(_SCRNCell(alpha=0.95))
    elif paper_id in {"antisymmetric-rnn"}:
        encoder = _AntisymmetricRNNEncoder()
    elif paper_id in {"cornn"}:
        encoder = _CoRNNEncoder()
    elif paper_id in {"unicornn"}:
        encoder = _UnICORNNEncoder()
    elif paper_id in {"lem"}:
        encoder = _LEMEncoder()
    elif paper_id in {"tau-gru"}:

        class _TauGRU(_SeqEncoder):
            def __init__(self, *, delay: int = 2) -> None:
                super().__init__()
                self.d = int(delay)
                self.cell = _GRUVariantCell(variant=0)
                self.w = nn.Parameter(torch.zeros((self.d,), dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                hist = [
                    torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                    for _ in range(self.d + 1)
                ]
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    # weighted delayed feedback (simple)
                    fb = 0.0
                    for k in range(1, self.d + 1):
                        fb = fb + torch.sigmoid(self.w[k - 1]) * hist[(t - k) % (self.d + 1)]
                    h_prev = hist[t % (self.d + 1)] + fb
                    h_new, _ = self.cell.step(x_t, h_prev, t=t)
                    hist[t % (self.d + 1)] = h_new
                    outs.append(h_new.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _TauGRU(delay=2)
    elif paper_id in {"dg-rnn"}:

        class _DGRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.core = _RNNCell()
                self.mask = nn.Linear(1 + hid, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_new, _ = self.core.step(x_t, h_t, t=t)
                    m = torch.sigmoid(self.mask(torch.cat([x_t, h_t], dim=-1)))
                    h_t = m * h_new + (1.0 - m) * h_t
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _DGRNN()
    elif paper_id in {"star"}:

        class _STAR(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2z = nn.Linear(1, hid, bias=True)
                self.h2z = nn.Linear(hid, hid, bias=False)
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)
                self.gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                g = torch.clamp(self.gain, 0.1, 10.0)
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_t))
                    h_hat = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                    h_t = (1.0 - z) * h_t + z * (g * h_hat)
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _STAR()
    elif paper_id in {"strongly-typed-rnn"}:

        class _STRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.x2f = nn.Linear(1, hid, bias=True)
                self.h2f = nn.Linear(hid, hid, bias=False)
                self.x2s = nn.Linear(1, hid, bias=True)
                self.h2s = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                s = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                f = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    f = torch.tanh(self.x2f(x_t) + self.h2f(f))
                    u = torch.sigmoid(self.x2s(x_t) + self.h2s(s))
                    s = (1.0 - u) * s + u * f
                    outs.append(s.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _STRNN()
    elif paper_id in {"multiplicative-lstm"}:

        class _mLSTM(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.wmx = nn.Linear(1, hid, bias=True)
                self.wmh = nn.Linear(hid, hid, bias=False)
                self.x2h = nn.Linear(1, 4 * hid, bias=True)
                self.m2h = nn.Linear(hid, 4 * hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    m_t = self.wmx(x_t) * self.wmh(h_t)
                    gates = self.x2h(x_t) + self.m2h(m_t)
                    i, f, g, o = gates.chunk(4, dim=-1)
                    i = torch.sigmoid(i)
                    f = torch.sigmoid(f)
                    g = torch.tanh(g)
                    o = torch.sigmoid(o)
                    c_t = f * c_t + i * g
                    h_t = o * torch.tanh(c_t)
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _mLSTM()
    elif paper_id in {"brc", "nbrc"}:

        class _BRC(_SeqEncoder):
            def __init__(self, *, neuromod: bool) -> None:
                super().__init__()
                self.neuromod = bool(neuromod)
                self.x2z = nn.Linear(1, hid, bias=True)
                self.h2z = nn.Linear(hid, hid, bias=False)
                self.x2u = nn.Linear(1, hid, bias=True)
                self.h2u = nn.Linear(hid, hid, bias=False)
                self.x2h = nn.Linear(1, hid, bias=True)
                self.h2h = nn.Linear(hid, hid, bias=False)
                self.mod = nn.Linear(hid, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    z = torch.sigmoid(self.x2z(x_t) + self.h2z(h_t))
                    u = torch.sigmoid(self.x2u(x_t) + self.h2u(h_t))
                    h_hat = torch.tanh(self.x2h(x_t) + self.h2h(h_t))
                    if self.neuromod:
                        u = u * torch.sigmoid(self.mod(h_t))
                    # bistable-ish mix: preserve h when u small, switch when u large
                    h_t = (1.0 - z) * (u * h_t + (1.0 - u) * h_t.sign()) + z * h_hat
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _BRC(neuromod=(paper_id == "nbrc"))
    elif paper_id in {"residual-rnn"}:
        encoder = _ResidualRNNEncoder()
    elif paper_id in {"orthogonal-rnn"}:
        encoder = _OrthogonalRNNEncoder()
    elif paper_id in {"unitary-rnn"}:
        encoder = _UnitaryCayleyRNNEncoder()
    elif paper_id in {"eunn"}:
        encoder = _EUNNEncoder()
    elif paper_id in {"goru"}:
        encoder = _GORUEncoder()
    elif paper_id in {"ode-rnn"}:
        encoder = _ODERNNEncoder()
    elif paper_id in {"neural-cde"}:
        encoder = _NeuralCDEEncoder()
    elif paper_id in {"echo-state-network"}:
        encoder = _ESNEncoder(spectral_radius=float(spectral_radius), leak=float(leak))
    elif paper_id in {"deep-esn"}:
        encoder = _DeepESNEncoder(depth=3, spectral_radius=float(spectral_radius), leak=float(leak))
    elif paper_id in {"liquid-state-machine"}:

        class _LSM(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.esn = _ESNEncoder(spectral_radius=float(spectral_radius), leak=float(leak))
                self.th = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
                self.k = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                seq = self.esn(xb)
                # spiking-like nonlinearity
                return torch.sigmoid(self.k * (seq - self.th))

        encoder = _LSM()
    elif paper_id in {"conceptor-rnn"}:
        encoder = _ConceptorESNEncoder(spectral_radius=float(spectral_radius), leak=float(leak))
    elif paper_id in {"deep-ar"}:
        # DeepAR-style: train a 1-step probabilistic RNN, decode recursively with predictive mean.
        Y_next = Y[:, :1].reshape(Y.shape[0], 1)

        class _DeepAR(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                rnn_drop = float(drop) if int(layers) > 1 else 0.0
                self.rnn = _StackedScanEncoder(
                    [_GRUCell(in_dim=(1 if i == 0 else hid)) for i in range(int(layers))],
                    dropout=rnn_drop,
                )
                self.head = nn.Linear(hid, 2, bias=True)

            def forward(self, xb: Any) -> Any:
                out = self.rnn(xb)
                last = out[:, -1, :]
                return self.head(last)  # (B,2) = (mu, raw_sigma)

        def _gaussian_nll(pred: Any, yb: Any) -> Any:
            mu = pred[:, 0:1]
            sigma = F.softplus(pred[:, 1:2]) + 1e-3
            z = (yb - mu) / sigma
            return (0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * (z**2)).mean()

        model = _DeepAR()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss="mse",
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(
            model, X_seq, Y_next, cfg=cfg, device=str(device), loss_fn_override=_gaussian_nll
        )

        hist = x_work[-lag_count:].astype(float, copy=True)
        preds: list[float] = []
        dev = torch.device(str(device))
        with torch.no_grad():
            for _ in range(h):
                feat = hist.reshape(1, lag_count, 1)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
                out = model(feat_t)
                mu = float(out[:, 0].detach().cpu().item())
                preds.append(mu)
                hist = np.concatenate([hist[1:], np.array([mu], dtype=float)], axis=0)

        yhat = np.asarray(preds, dtype=float)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"mqrnn"}:
        quantiles = (0.1, 0.5, 0.9)

        class _MQRNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                enc_layers = max(1, int(layers))
                rnn_drop = float(drop) if enc_layers > 1 else 0.0
                self.enc = _StackedScanEncoder(
                    [_GRUCell(in_dim=(1 if i == 0 else hid)) for i in range(int(enc_layers))],
                    dropout=rnn_drop,
                )
                self.head = nn.Linear(hid, h * len(quantiles), bias=True)

            def forward(self, xb: Any) -> Any:
                out = self.enc(xb)
                ctx = out[:, -1, :]
                q = self.head(ctx).reshape(xb.shape[0], h, len(quantiles))
                return q

        def _pinball(pred: Any, yb: Any) -> Any:
            # pred: (B,h,Q), yb: (B,h)
            y = yb.unsqueeze(-1)
            loss_v = 0.0
            for qi, q in enumerate(quantiles):
                e = y - pred[:, :, qi : qi + 1]
                loss_v = loss_v + torch.maximum(q * e, (q - 1.0) * e).mean()
            return loss_v / float(len(quantiles))

        model = _MQRNN()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss="mse",
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device), loss_fn_override=_pinball)

        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        dev = torch.device(str(device))
        with torch.no_grad():
            q_pred = model(torch.tensor(feat, dtype=torch.float32, device=dev))
            median = q_pred[:, :, 1].detach().cpu().numpy().reshape(-1)
        yhat = np.asarray(median, dtype=float)
        if bool(normalize):
            yhat = yhat * std + mean
        return yhat
    elif paper_id in {"deepstate"}:

        class _DeepStateLite(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = nn.Linear(3, 2, bias=True)
                self.head = nn.Linear(2, h, bias=True)

            def forward(self, xb: Any) -> Any:
                # xb: (B,T,1) -> adaptive level/trend smoothing
                B, T, _ = xb.shape
                lvl = xb[:, 0, :].expand(int(B), 1)
                tr = torch.zeros((int(B), 1), device=xb.device, dtype=xb.dtype)
                for t in range(1, int(T)):
                    x_t = xb[:, t, :]
                    err = x_t - lvl
                    g = torch.sigmoid(self.gate(torch.cat([x_t, lvl, tr], dim=-1)))
                    alpha = g[:, 0:1]
                    beta = g[:, 1:2]
                    lvl = lvl + tr + alpha * err
                    tr = tr + beta * err
                # horizon features: [lvl, tr]
                feat = torch.cat([lvl, tr], dim=-1)
                return self.head(feat)

        encoder = None  # handled below
        model = _DeepStateLite()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))
        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        dev = torch.device(str(device))
        with torch.no_grad():
            yhat_t = (
                model(torch.tensor(feat, dtype=torch.float32, device=dev))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"lstnet"}:
        # LSTNet-style (lite for univariate): CNN features + GRU encoder + skip GRU + AR highway.
        skip = 2
        ar_window = min(lag_count, 24)

        class _LSTNetLite(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                ks = int(max(1, kernel_size))
                self.conv = nn.Conv1d(1, hid, kernel_size=ks, padding=ks - 1)
                self.gru = _StackedScanEncoder([_GRUCell(in_dim=hid)], dropout=0.0)
                self.skip_gru = _StackedScanEncoder([_GRUCell(in_dim=hid)], dropout=0.0)
                self.fuse = nn.Sequential(
                    nn.Linear(2 * hid, hid, bias=True),
                    nn.ReLU(),
                    nn.Dropout(float(drop)),
                )
                self.head = nn.Linear(hid, h, bias=True)
                self.ar = nn.Linear(int(ar_window), h, bias=True)

            def forward(self, xb: Any) -> Any:
                # xb: (B,T,1)
                B, T, _ = xb.shape
                x1 = xb.transpose(1, 2)  # (B,1,T)
                feat = torch.relu(self.conv(x1)[:, :, :T]).transpose(1, 2)  # (B,T,H)

                out = self.gru(feat)
                main = out[:, -1, :]

                skip_feat = feat[:, :: int(skip), :]
                out_s = self.skip_gru(skip_feat)
                sk = out_s[:, -1, :]

                ctx = self.fuse(torch.cat([main, sk], dim=-1))
                y_rnn = self.head(ctx)

                y = xb.squeeze(-1)
                y_ar = self.ar(y[:, -int(ar_window) :])
                return y_rnn + y_ar

        model = _LSTNetLite()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        with torch.no_grad():
            feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
            yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"esrnn"}:
        # ESRNN-style hybrid (lite): Holt baseline + RNN residual adjustment.
        alpha_init = 0.3
        beta_init = 0.1

        def _logit(p: float) -> float:
            p2 = min(max(float(p), 1e-4), 1.0 - 1e-4)
            return math.log(p2 / (1.0 - p2))

        alpha_logit_init = _logit(alpha_init)
        beta_logit_init = _logit(beta_init)

        class _ESRNNDirect(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init, dtype=torch.float32))
                self.beta_logit = nn.Parameter(torch.tensor(beta_logit_init, dtype=torch.float32))
                enc_layers = max(1, int(layers))
                rnn_drop = float(drop) if enc_layers > 1 else 0.0
                self.rnn = _StackedScanEncoder(
                    [_GRUCell(in_dim=(1 if i == 0 else hid)) for i in range(int(enc_layers))],
                    dropout=rnn_drop,
                )
                self.norm = nn.LayerNorm(hid)
                self.head = nn.Linear(hid, h)

            def forward(self, xb: Any) -> Any:
                if xb.ndim == 2:
                    xb = xb.unsqueeze(-1)
                y = xb.squeeze(-1)  # (B,T)
                B = int(y.shape[0])
                T = int(y.shape[1])

                alpha = torch.sigmoid(self.alpha_logit)
                beta = torch.sigmoid(self.beta_logit)

                level = y[:, 0]
                if T >= 2:
                    tr = y[:, 1] - y[:, 0]
                else:
                    tr = torch.zeros((B,), device=y.device, dtype=y.dtype)

                levels: list[Any] = [level]
                trends: list[Any] = [tr]
                for t in range(1, T):
                    yt = y[:, t]
                    level_new = alpha * yt + (1.0 - alpha) * (level + tr)
                    tr_new = beta * (level_new - level) + (1.0 - beta) * tr
                    level, tr = level_new, tr_new
                    levels.append(level)
                    trends.append(tr)

                level_seq = torch.stack(levels, dim=1)  # (B,T)
                trend_seq = torch.stack(trends, dim=1)  # (B,T)

                fitted = torch.empty_like(y)
                fitted[:, 0] = level_seq[:, 0]
                if T > 1:
                    fitted[:, 1:] = level_seq[:, :-1] + trend_seq[:, :-1]

                resid = (y - fitted).unsqueeze(-1)  # (B,T,1)

                steps = torch.arange(1, int(h) + 1, device=y.device, dtype=y.dtype).reshape(1, -1)
                baseline = level_seq[:, -1].unsqueeze(1) + steps * trend_seq[:, -1].unsqueeze(1)

                out = self.rnn(resid)
                last = self.norm(out[:, -1, :])
                delta = self.head(last)
                return baseline + delta

        model = _ESRNNDirect()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        with torch.no_grad():
            feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
            yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {
        "memory-networks",
        "end-to-end-memory-networks",
        "dynamic-memory-networks",
        "pointer-network",
        "pointer-sentinel-mixture",
        "copynet",
    }:
        hp = int(max(1, hops))
        if paper_id == "memory-networks":
            encoder = _MemoryNetEncoder(hops=1, attn_hidden=int(attn_hidden))
        elif paper_id == "end-to-end-memory-networks":
            encoder = _MemoryNetEncoder(hops=hp, attn_hidden=int(attn_hidden))
        elif paper_id == "dynamic-memory-networks":
            encoder = _MemoryNetEncoder(hops=hp + 1, attn_hidden=int(attn_hidden))
        else:
            # pointer/copy models: use dot attention pooling as the "pointer"
            base = _MemoryNetEncoder(hops=1, attn_hidden=int(attn_hidden))

            class _PointerWrap(_SeqEncoder):
                def __init__(self, base_enc: Any, *, sentinel: bool, copy_mix: bool) -> None:
                    super().__init__()
                    self.base = base_enc
                    self.sentinel = bool(sentinel)
                    self.copy_mix = bool(copy_mix)
                    self.gen = nn.Linear(hid, hid, bias=True)
                    self.s = nn.Parameter(torch.zeros((hid,), dtype=torch.float32))
                    self.mix = nn.Linear(hid, 1, bias=True)

                def forward(self, xb: Any) -> Any:
                    mem_seq = self.base(xb)  # repeated q: (B,T,H)
                    # pointer over original inputs (as values)
                    ctx = _DotAttentionPool()(mem_seq)  # (B,H)
                    if self.sentinel:
                        g = torch.sigmoid(self.mix(ctx))
                        ctx = g * ctx + (1.0 - g) * self.s.unsqueeze(0)
                    if self.copy_mix:
                        gen = torch.tanh(self.gen(ctx))
                        m = torch.sigmoid(self.mix(gen))
                        ctx = m * gen + (1.0 - m) * ctx
                    return ctx.unsqueeze(1).expand(xb.shape[0], xb.shape[1], hid)

            encoder = _PointerWrap(
                base,
                sentinel=(paper_id == "pointer-sentinel-mixture"),
                copy_mix=(paper_id == "copynet"),
            )
    elif paper_id in {"neural-turing-machine", "differentiable-neural-computer"}:
        encoder = _NTMEncoder(slots=int(memory_slots), mem_dim=int(memory_dim))
    elif paper_id in {"rnn-transducer"}:
        # RNN-T-lite: encoder RNN + prediction RNN unrolled for horizon, no teacher forcing.
        class _RNNT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                enc_layers = max(1, int(layers))
                rnn_drop = float(drop) if enc_layers > 1 else 0.0
                self.enc = _StackedScanEncoder(
                    [_GRUCell(in_dim=(1 if i == 0 else hid)) for i in range(int(enc_layers))],
                    dropout=rnn_drop,
                )
                self.pred = _GRUCell(in_dim=1)
                self.joint = nn.Linear(2 * hid, hid, bias=True)
                self.out = nn.Linear(hid, 1, bias=True)

            def forward(self, xb: Any) -> Any:
                enc_out = self.enc(xb)
                ctx = enc_out[:, -1, :]
                y_prev = xb[:, -1, :]
                h_p = torch.zeros((xb.shape[0], hid), device=xb.device, dtype=xb.dtype)
                ys = []
                for t in range(int(h)):
                    h_p, _ = self.pred.step(y_prev, h_p, t=t)
                    j = torch.tanh(self.joint(torch.cat([ctx, h_p], dim=-1)))
                    y_next = self.out(j)
                    ys.append(y_next)
                    y_prev = y_next
                return torch.cat(ys, dim=1)

        model = _RNNT()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))
        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        dev = torch.device(str(device))
        with torch.no_grad():
            yhat_t = (
                model(torch.tensor(feat, dtype=torch.float32, device=dev))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"seq2seq", "rnn-encoder-decoder", "bahdanau-attention"}:
        use_bahdanau = paper_id == "bahdanau-attention"

        class _BahdanauAttention(nn.Module):
            def __init__(self, *, attn_hidden: int) -> None:
                super().__init__()
                a = int(attn_hidden)
                if a <= 0:
                    raise ValueError("attn_hidden must be >= 1")
                self.W_enc = nn.Linear(hid, a, bias=False)
                self.W_dec = nn.Linear(hid, a, bias=False)
                self.v = nn.Linear(a, 1, bias=False)

            def forward(self, enc_out: Any, dec_h: Any) -> Any:
                # enc_out: (B,T,H), dec_h: (B,H)
                e = self.W_enc(enc_out) + self.W_dec(dec_h).unsqueeze(1)  # (B,T,A)
                scores = self.v(torch.tanh(e)).squeeze(-1)  # (B,T)
                w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
                return torch.sum(w * enc_out, dim=1)  # (B,H)

        class _Seq2Seq(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                enc_layers = max(1, int(layers))
                self.enc_cells = nn.ModuleList(
                    [_LSTMCell(in_dim=(1 if i == 0 else hid)) for i in range(int(enc_layers))]
                )
                self.attn = (
                    _BahdanauAttention(attn_hidden=int(attn_hidden)) if use_bahdanau else None
                )
                self.dec_cell = _LSTMCell(in_dim=1 + hid)
                self.out = nn.Linear(hid, 1, bias=True)
                self.drop = float(drop)

            def _encode(self, xb: Any) -> tuple[Any, tuple[Any, Any]]:
                B, T, _ = xb.shape
                states = [
                    cell.init_state(int(B), device=xb.device, dtype=xb.dtype)
                    for cell in self.enc_cells
                ]
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    for i, cell in enumerate(self.enc_cells):
                        states[i], h_t = cell.step(x_t, states[i], t=t)
                        x_t = h_t
                        if i < len(self.enc_cells) - 1 and self.drop > 0.0 and self.training:
                            x_t = F.dropout(x_t, p=self.drop, training=True)
                    outs.append(x_t.unsqueeze(1))
                enc_out = torch.cat(outs, dim=1)
                h_last, c_last = states[-1]
                return enc_out, (h_last, c_last)

            def forward(
                self, xb: Any, yb: Any | None = None, *, teacher_forcing_ratio: float
            ) -> Any:
                B, T, _ = xb.shape
                enc_out, (h_t, c_t) = self._encode(xb)

                y_prev = xb[:, -1, :]  # (B,1)
                ys: list[Any] = []
                tf = float(teacher_forcing_ratio)
                for t in range(int(h)):
                    if self.attn is None:
                        ctx = enc_out[:, -1, :]
                    else:
                        ctx = self.attn(enc_out, h_t)
                    inp = torch.cat([y_prev, ctx], dim=-1)
                    (h_t, c_t), h_out = self.dec_cell.step(inp, (h_t, c_t), t=t)
                    y_next = self.out(h_out)
                    ys.append(y_next)

                    if yb is not None and tf > 0.0:
                        use_teacher = torch.rand((int(B), 1), device=xb.device) < tf
                        y_prev = torch.where(use_teacher, yb[:, t : t + 1], y_next)
                    else:
                        y_prev = y_next

                return torch.cat(ys, dim=1)  # (B,h)

        def _make_loss_fn(nn: Any, loss: str) -> Any:
            name = str(loss).lower().strip()
            if name in {"mse", ""}:
                return nn.MSELoss()
            if name in {"mae", "l1"}:
                return nn.L1Loss()
            if name in {"huber", "smoothl1"}:
                return nn.SmoothL1Loss()
            raise ValueError("loss must be one of: mse, mae, huber")

        def _train_seq2seq(
            model: Any,
            X: np.ndarray,
            Y: np.ndarray,
            *,
            cfg: TorchTrainConfig,
            device: str,
            teacher_forcing_start: float,
            teacher_forcing_final: float,
        ) -> Any:
            if cfg.epochs <= 0:
                raise ValueError("epochs must be >= 1")
            if cfg.lr <= 0.0:
                raise ValueError("lr must be > 0")
            if cfg.batch_size <= 0:
                raise ValueError("batch_size must be >= 1")
            if cfg.patience <= 0:
                raise ValueError("patience must be >= 1")
            if float(cfg.val_split) < 0.0 or float(cfg.val_split) >= 0.5:
                raise ValueError("val_split must be in [0, 0.5)")
            if float(cfg.grad_clip_norm) < 0.0:
                raise ValueError("grad_clip_norm must be >= 0")

            tf0 = float(teacher_forcing_start)
            tf1 = float(teacher_forcing_final)
            if not (0.0 <= tf0 <= 1.0):
                raise ValueError("teacher_forcing_start must be in [0,1]")
            if not (0.0 <= tf1 <= 1.0):
                raise ValueError("teacher_forcing_final must be in [0,1]")

            torch.manual_seed(int(cfg.seed))

            dev = torch.device(str(device))
            if dev.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("device='cuda' requested but CUDA is not available")

            model = model.to(dev)

            X_t = torch.tensor(X, dtype=torch.float32, device=dev)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=dev)

            n = int(X_t.shape[0])
            val_n = 0
            if float(cfg.val_split) > 0.0 and n >= 5:
                val_n = max(1, int(round(float(cfg.val_split) * n)))
                val_n = min(val_n, n - 1)

            if val_n > 0:
                train_end = n - val_n
                X_train, Y_train = X_t[:train_end], Y_t[:train_end]
                X_val, Y_val = X_t[train_end:], Y_t[train_end:]
            else:
                X_train, Y_train = X_t, Y_t
                X_val, Y_val = None, None

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, Y_train),
                batch_size=int(cfg.batch_size),
                shuffle=True,
            )
            val_loader = (
                None
                if X_val is None
                else torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_val, Y_val),
                    batch_size=int(cfg.batch_size),
                    shuffle=False,
                )
            )

            opt_name = str(cfg.optimizer).lower().strip()
            if opt_name in {"adam", ""}:
                opt = torch.optim.Adam(
                    model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
                )
            elif opt_name == "adamw":
                opt = torch.optim.AdamW(
                    model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
                )
            elif opt_name == "sgd":
                opt = torch.optim.SGD(
                    model.parameters(),
                    lr=float(cfg.lr),
                    momentum=float(cfg.momentum),
                    weight_decay=float(cfg.weight_decay),
                )
            else:
                raise ValueError("optimizer must be one of: adam, adamw, sgd")

            loss_fn = _make_loss_fn(nn, cfg.loss)

            sched_name = str(cfg.scheduler).lower().strip()
            if sched_name in {"none", ""}:
                sched = None
            elif sched_name == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.epochs))
            elif sched_name == "step":
                sched = torch.optim.lr_scheduler.StepLR(
                    opt,
                    step_size=int(cfg.scheduler_step_size),
                    gamma=float(cfg.scheduler_gamma),
                )
            else:
                raise ValueError("scheduler must be one of: none, cosine, step")

            best_loss = float("inf")
            best_state: dict[str, Any] | None = None
            bad_epochs = 0

            for epoch in range(int(cfg.epochs)):
                if int(cfg.epochs) == 1:
                    tf = tf1
                else:
                    t = float(epoch) / float(int(cfg.epochs) - 1)
                    tf = (1.0 - t) * tf0 + t * tf1

                model.train()
                total = 0.0
                count = 0
                for xb, yb in train_loader:
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb, yb, teacher_forcing_ratio=float(tf))
                    loss_v = loss_fn(pred, yb)
                    loss_v.backward()
                    if float(cfg.grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=float(cfg.grad_clip_norm)
                        )
                    opt.step()
                    total += float(loss_v.detach().cpu().item()) * int(xb.shape[0])
                    count += int(xb.shape[0])
                train_loss = total / max(1, count)

                if val_loader is not None:
                    model.eval()
                    v_total = 0.0
                    v_count = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            pred = model(xb, yb, teacher_forcing_ratio=0.0)
                            v_loss = loss_fn(pred, yb)
                            v_total += float(v_loss.detach().cpu().item()) * int(xb.shape[0])
                            v_count += int(xb.shape[0])
                    monitor = v_total / max(1, v_count)
                else:
                    monitor = train_loss

                if float(monitor) + 1e-12 < best_loss:
                    best_loss = float(monitor)
                    bad_epochs = 0
                    if bool(cfg.restore_best):
                        best_state = {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        }
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(cfg.patience):
                        break

                if sched is not None:
                    sched.step()

            if bool(cfg.restore_best) and best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            return model

        model = _Seq2Seq()
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_seq2seq(
            model,
            X_seq,
            Y,
            cfg=cfg,
            device=str(device),
            teacher_forcing_start=0.5,
            teacher_forcing_final=0.0,
        )

        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        dev = torch.device(str(device))
        with torch.no_grad():
            feat_t = torch.tensor(feat, dtype=torch.float32, device=dev)
            yhat_t = (
                model(feat_t, None, teacher_forcing_ratio=0.0).detach().cpu().numpy().reshape(-1)
            )

        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"luong-attention"}:
        # Dot-product attention pooling over an LSTM encoder.
        rnn_drop = float(drop) if int(layers) > 1 else 0.0
        encoder = _StackedScanEncoder(
            [_LSTMCell(in_dim=(1 if i == 0 else hid)) for i in range(int(layers))],
            dropout=rnn_drop,
        )
    elif paper_id in {"neural-stack", "neural-queue", "neural-ram"}:
        # Data-structure-inspired memory via restricted NTM addressing.
        encoder = _NTMEncoder(slots=int(memory_slots), mem_dim=int(memory_dim))
    elif paper_id in {"recurrent-attention-model"}:

        class _RAM(_SeqEncoder):
            def __init__(self, *, glimpses: int = 4) -> None:
                super().__init__()
                self.g = int(glimpses)
                if self.g <= 0:
                    raise ValueError("glimpses must be >= 1")
                self.emb = nn.Linear(1, hid, bias=True)
                self.ctrl = _GRUCell(in_dim=hid)
                self.attn = nn.Linear(hid, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                mem = self.emb(xb)  # (B,T,H)
                B, T, _ = mem.shape
                h_c = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                for gi in range(self.g):
                    q = self.attn(h_c).unsqueeze(2)
                    scores = torch.bmm(mem, q).squeeze(2) / max(1.0, float(hid) ** 0.5)
                    w = torch.softmax(scores, dim=1).unsqueeze(-1)
                    glimpse = torch.sum(w * mem, dim=1)
                    h_c, _ = self.ctrl.step(glimpse, h_c, t=int(gi))
                return h_c.unsqueeze(1).expand(int(B), int(T), hid)

        encoder = _RAM(glimpses=4)
    elif paper_id in {"convlstm", "convgru", "trajgru"}:
        # Conv*-lite: causal Conv1d feature extractor + recurrent core.
        ks = int(max(1, kernel_size))
        conv = nn.Conv1d(1, hid, kernel_size=ks, padding=ks - 1)
        if paper_id == "convgru" or paper_id == "trajgru":
            core = _StackedScanEncoder([_GRUCell(in_dim=hid)], dropout=0.0)
        else:
            core = _StackedScanEncoder([_LSTMCell(in_dim=hid)], dropout=0.0)

        class _ConvRNN(_SeqEncoder):
            def __init__(self, conv: Any, core: Any) -> None:
                super().__init__()
                self.conv = conv
                self.core = core

            def forward(self, xb: Any) -> Any:
                x1 = xb.transpose(1, 2)  # (B,1,T)
                feat = self.conv(x1)[:, :, : xb.shape[1]].transpose(1, 2)  # (B,T,H)
                return self.core(feat)

        encoder = _ConvRNN(conv, core)
    elif paper_id in {"predrnn", "predrnn-plus-plus"}:

        class _PredRNN(nn.Module):
            def __init__(self, *, highway: bool) -> None:
                super().__init__()
                self.highway = bool(highway)
                self.x2h = nn.Linear(1, 4 * hid, bias=True)
                self.h2h = nn.Linear(hid, 4 * hid, bias=False)
                self.m2h = nn.Linear(hid, 4 * hid, bias=False)
                self.head = nn.Linear(hid, h, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                c_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                m_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    gates = self.x2h(x_t) + self.h2h(h_t) + self.m2h(m_t)
                    i, f, g, o = gates.chunk(4, dim=-1)
                    i = torch.sigmoid(i)
                    f = torch.sigmoid(f)
                    g = torch.tanh(g)
                    o = torch.sigmoid(o)
                    c_new = f * c_t + i * g
                    m_new = f * m_t + (1.0 - f) * c_new
                    h_new = o * torch.tanh(c_new + m_new)
                    if self.highway:
                        h_t = h_t + 0.1 * h_new
                    else:
                        h_t = h_new
                    c_t = c_new
                    m_t = m_new
                return self.head(h_t)

        model = _PredRNN(highway=(paper_id == "predrnn-plus-plus"))
        cfg = TorchTrainConfig(
            epochs=int(epochs),
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            seed=int(seed),
            patience=int(patience),
            loss=str(loss),
            val_split=float(val_split),
            grad_clip_norm=float(grad_clip_norm),
            optimizer=str(optimizer),
            momentum=float(momentum),
            scheduler=str(scheduler),
            scheduler_step_size=int(scheduler_step_size),
            scheduler_gamma=float(scheduler_gamma),
            restore_best=bool(restore_best),
        )
        model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))
        feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
        dev = torch.device(str(device))
        with torch.no_grad():
            yhat_t = (
                model(torch.tensor(feat, dtype=torch.float32, device=dev))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
        yhat = yhat_t.astype(float, copy=False)
        if bool(normalize):
            yhat = yhat * std + mean
        return np.asarray(yhat, dtype=float)
    elif paper_id in {"dcrnn"}:
        # DCRNN-lite: GRU with a diffusion-style mixing matrix on hidden state.
        class _DCRNN(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.cell = _GRUCell(in_dim=1)
                self.mix = nn.Linear(hid, hid, bias=False)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                h_t = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    h_t, _ = self.cell.step(x_t, h_t, t=t)
                    h_t = torch.tanh(self.mix(h_t))
                    outs.append(h_t.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _DCRNN()
    elif paper_id in {"structural-rnn"}:

        class _Structural(_SeqEncoder):
            def __init__(self) -> None:
                super().__init__()
                self.node = _RNNCell(in_dim=hid)
                self.edge = _RNNCell()
                self.fuse = nn.Linear(2 * hid, hid, bias=True)

            def forward(self, xb: Any) -> Any:
                B, T, _ = xb.shape
                hn = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                he = torch.zeros((int(B), hid), device=xb.device, dtype=xb.dtype)
                outs = []
                for t in range(int(T)):
                    x_t = xb[:, t, :]
                    he, _ = self.edge.step(x_t, he, t=t)
                    fuse_in = torch.tanh(self.fuse(torch.cat([x_t.expand_as(hn), he], dim=-1)))
                    hn, _ = self.node.step(fuse_in, hn, t=t)
                    outs.append(hn.unsqueeze(1))
                return torch.cat(outs, dim=1)

        encoder = _Structural()
    else:
        # Should be unreachable if _PAPER_DEFS and selection stay in sync.
        raise RuntimeError(f"Unhandled rnnpaper architecture: {paper_id!r}")

    pool = "last"
    if paper_id in {"luong-attention", "pointer-network"}:
        pool = "dot"
    if paper_id in {"bahdanau-attention"}:
        pool = "additive"

    model = _RNNPaperNet(encoder=encoder, pool=pool)

    cfg = TorchTrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        seed=int(seed),
        patience=int(patience),
        loss=str(loss),
        val_split=float(val_split),
        grad_clip_norm=float(grad_clip_norm),
        optimizer=str(optimizer),
        momentum=float(momentum),
        scheduler=str(scheduler),
        scheduler_step_size=int(scheduler_step_size),
        scheduler_gamma=float(scheduler_gamma),
        restore_best=bool(restore_best),
    )
    model = _train_loop(model, X_seq, Y, cfg=cfg, device=str(device))

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
