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


def _positional_encoding_sincos(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((int(seq_len), int(d_model)), dtype=float)
    position = np.arange(int(seq_len), dtype=float).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, int(d_model), 2, dtype=float) * (-math.log(10000.0) / float(d_model))
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def _check_dropout(p: float) -> float:
    drop = float(p)
    if not (0.0 <= drop < 1.0):
        raise ValueError("dropout must be in [0, 1)")
    return drop


@dataclass(frozen=True)
class XFormerConfig:
    lags: int
    horizon: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    attn: str
    pos_emb: str
    norm: str
    ffn: str
    local_window: int
    performer_features: int
    linformer_k: int
    nystrom_landmarks: int
    horizon_tokens: str
    revin: bool
    residual_gating: bool
    drop_path: float


def torch_xformer_direct_forecast(
    train: Any,
    horizon: int,
    *,
    lags: int = 96,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    attn: str = "full",
    pos_emb: str = "learned",
    norm: str = "layer",
    ffn: str = "gelu",
    local_window: int = 16,
    bigbird_random_k: int = 8,
    performer_features: int = 64,
    linformer_k: int = 32,
    nystrom_landmarks: int = 16,
    reformer_bucket_size: int = 8,
    reformer_n_hashes: int = 1,
    probsparse_top_u: int = 32,
    autocorr_top_k: int = 4,
    horizon_tokens: str = "zeros",
    revin: bool = False,
    residual_gating: bool = False,
    drop_path: float = 0.0,
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
    Configurable Torch Transformer-family forecaster (direct multi-horizon).

    This model trains on windows of length (lags + horizon):
      - past tokens contain observed y
      - future tokens are zeros or learned "query" tokens

    It supports multiple attention approximations:
      attn = full | local | logsparse | longformer | bigbird | performer | linformer | nystrom | probsparse | autocorr | reformer
    """
    torch = _require_torch()
    nn = torch.nn
    F = torch.nn.functional

    x = _as_1d_float_array(train)
    h = int(horizon)
    lag_count = int(lags)
    if h <= 0:
        raise ValueError("horizon must be >= 1")
    if lag_count <= 0:
        raise ValueError("lags must be >= 1")

    d = int(d_model)
    heads = int(nhead)
    if d <= 0:
        raise ValueError("d_model must be >= 1")
    if heads <= 0:
        raise ValueError("nhead must be >= 1")
    if d % heads != 0:
        raise ValueError("d_model must be divisible by nhead")
    if int(num_layers) <= 0:
        raise ValueError("num_layers must be >= 1")
    if int(dim_feedforward) <= 0:
        raise ValueError("dim_feedforward must be >= 1")

    drop = _check_dropout(float(dropout))
    drop_path_f = _check_dropout(float(drop_path))

    attn_s = str(attn).lower().strip()
    pos_s = str(pos_emb).lower().strip()
    norm_s = str(norm).lower().strip()
    ffn_s = str(ffn).lower().strip()
    horizon_tokens_s = str(horizon_tokens).lower().strip()

    if attn_s not in {
        "full",
        "local",
        "logsparse",
        "longformer",
        "bigbird",
        "performer",
        "linformer",
        "nystrom",
        "probsparse",
        "autocorr",
        "reformer",
    }:
        raise ValueError(
            "attn must be one of: full, local, logsparse, longformer, bigbird, performer, linformer, nystrom, probsparse, autocorr, reformer"
        )
    if pos_s not in {"learned", "sincos", "rope", "time2vec", "none"}:
        raise ValueError("pos_emb must be one of: learned, sincos, rope, time2vec, none")
    if norm_s not in {"layer", "rms"}:
        raise ValueError("norm must be one of: layer, rms")
    if ffn_s not in {"gelu", "swiglu"}:
        raise ValueError("ffn must be one of: gelu, swiglu")
    if horizon_tokens_s not in {"zeros", "learned"}:
        raise ValueError("horizon_tokens must be one of: zeros, learned")

    if int(local_window) <= 0:
        raise ValueError("local_window must be >= 1")
    bigbird_k = int(bigbird_random_k)
    if bigbird_k < 0:
        raise ValueError("bigbird_random_k must be >= 0")
    if int(performer_features) <= 0:
        raise ValueError("performer_features must be >= 1")
    if int(linformer_k) <= 0:
        raise ValueError("linformer_k must be >= 1")
    if int(nystrom_landmarks) <= 0:
        raise ValueError("nystrom_landmarks must be >= 1")

    reformer_bs = int(reformer_bucket_size)
    if reformer_bs <= 0:
        raise ValueError("reformer_bucket_size must be >= 1")
    reformer_hashes = int(reformer_n_hashes)
    if reformer_hashes <= 0:
        raise ValueError("reformer_n_hashes must be >= 1")

    probs_u = int(probsparse_top_u)
    if probs_u <= 0:
        raise ValueError("probsparse_top_u must be >= 1")
    auto_k = int(autocorr_top_k)
    if auto_k <= 0:
        raise ValueError("autocorr_top_k must be >= 1")

    x_work = x
    mean = 0.0
    std = 1.0
    if bool(normalize):
        x_work, mean, std = _normalize_series(x_work)

    X, Y = _make_lagged_xy_multi(x_work, lags=lag_count, horizon=h)
    # Build sequence input with horizon tokens appended.
    seq_len = lag_count + h
    X_pad = np.concatenate([X, np.zeros((X.shape[0], h), dtype=float)], axis=1)
    X_seq = X_pad.reshape(X_pad.shape[0], seq_len, 1)

    head_dim = d // heads

    def _make_rmsnorm(dim: int) -> Any:
        class _RMSNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.ones((dim,), dtype=torch.float32))

            def forward(self, xb: Any) -> Any:
                denom = torch.sqrt(torch.mean(xb * xb, dim=-1, keepdim=True) + 1e-8)
                return (xb / denom) * self.scale

        return _RMSNorm()

    def _rope_apply(q: Any, k: Any) -> tuple[Any, Any]:
        # q,k: (B, H, L, D)
        L = int(q.shape[2])
        D = int(q.shape[3])
        if D % 2 != 0:
            return q, k
        half = D // 2
        pos = torch.arange(L, device=q.device, dtype=torch.float32).reshape(1, 1, L, 1)
        freq = torch.exp(
            torch.arange(half, device=q.device, dtype=torch.float32)
            * (-math.log(10000.0) / float(half))
        ).reshape(1, 1, 1, half)
        ang = pos * freq
        sin = torch.sin(ang)
        cos = torch.cos(ang)

        def _rot(xb: Any) -> Any:
            x1 = xb[..., :half]
            x2 = xb[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _rot(q), _rot(k)

    class _Time2Vec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w0 = nn.Parameter(torch.randn(1, 1, 1))
            self.b0 = nn.Parameter(torch.zeros(1, 1, 1))
            self.W = nn.Parameter(torch.randn(1, 1, d - 1) * 0.1)
            self.b = nn.Parameter(torch.zeros(1, 1, d - 1))

        def forward(self, t: Any) -> Any:
            # t: (1, L, 1)
            v0 = self.w0 * t + self.b0
            v1 = torch.sin(t * self.W + self.b)
            return torch.cat([v0, v1], dim=-1)

    class _SwiGLUFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(dim_feedforward)
            self.fc = nn.Linear(d, 2 * hidden)
            self.proj = nn.Linear(hidden, d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            x2 = self.fc(xb)
            a, b = x2.chunk(2, dim=-1)
            z = F.silu(a) * b
            z = self.drop(z)
            return self.proj(z)

    class _GELUFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(dim_feedforward)
            self.fc1 = nn.Linear(d, hidden)
            self.fc2 = nn.Linear(hidden, d)
            self.drop = nn.Dropout(p=drop)

        def forward(self, xb: Any) -> Any:
            z = F.gelu(self.fc1(xb))
            z = self.drop(z)
            return self.fc2(z)

    class _MultiheadSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)

            # Linformer projections
            if attn_s == "linformer":
                self.E = nn.Parameter(torch.randn(heads, int(linformer_k), seq_len) * 0.02)
                self.F = nn.Parameter(torch.randn(heads, int(linformer_k), seq_len) * 0.02)
            else:
                self.register_parameter("E", None)
                self.register_parameter("F", None)

            # Performer random features
            if attn_s == "performer":
                # Random matrix for feature map (fixed).
                W = torch.randn(heads, head_dim, int(performer_features)) / math.sqrt(
                    float(head_dim)
                )
                self.register_buffer("W", W, persistent=False)
            else:
                self.register_buffer("W", torch.empty(0), persistent=False)

            # Reformer LSH random projections
            if attn_s == "reformer":
                n_buckets = int(max(1, int(math.ceil(float(seq_len) / float(reformer_bs)))))
                R = torch.randn(int(reformer_hashes), heads, head_dim, n_buckets) / math.sqrt(
                    float(head_dim)
                )
                self.register_buffer("R", R, persistent=False)
            else:
                self.register_buffer("R", torch.empty(0), persistent=False)

            # BigBird random connections (precomputed for this fixed sequence length).
            self.register_buffer("bigbird_rand", torch.empty(0, dtype=torch.bool), persistent=False)
            if attn_s == "bigbird":
                L = int(seq_len)
                rng = np.random.default_rng(int(seed) + 1337)

                idx_np = np.arange(L, dtype=int)
                dist = np.abs(idx_np.reshape(-1, 1) - idx_np.reshape(1, -1))
                local_allowed_np = dist <= int(local_window)

                global_mask_np = np.zeros((L,), dtype=bool)
                if int(h) > 0:
                    global_mask_np[L - int(h) :] = True
                last_ctx = L - int(h) - 1
                if last_ctx >= 0:
                    global_mask_np[int(last_ctx)] = True

                base_allowed = (
                    local_allowed_np | global_mask_np.reshape(L, 1) | global_mask_np.reshape(1, L)
                )

                rand_allowed = np.zeros((L, L), dtype=bool)
                if int(bigbird_k) > 0:
                    for i in range(L):
                        cand = np.flatnonzero(~base_allowed[i])
                        if cand.size == 0:
                            continue
                        pick = int(min(int(bigbird_k), int(cand.size)))
                        chosen = rng.choice(cand, size=pick, replace=False)
                        rand_allowed[i, chosen] = True
                    rand_allowed |= rand_allowed.T

                self.bigbird_rand = torch.tensor(rand_allowed, dtype=torch.bool)

        def _split_heads(self, xb: Any) -> Any:
            B, L, _D = xb.shape
            return xb.view(B, L, heads, head_dim).transpose(1, 2)  # (B,H,L,dh)

        def _merge_heads(self, xb: Any) -> Any:
            B, Hh, L, dh = xb.shape
            return xb.transpose(1, 2).contiguous().view(B, L, Hh * dh)

        def forward(self, xb: Any) -> Any:
            B, L, _ = xb.shape
            qkv = self.qkv(xb)
            q, k, v = qkv.chunk(3, dim=-1)
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)

            if pos_s == "rope":
                q, k = _rope_apply(q, k)

            scale = 1.0 / math.sqrt(float(head_dim))

            if attn_s == "performer":
                # Random-feature (positive) map; stable and fast.
                W = self.W  # (H,dh,m)
                # (B,H,L,m)
                qf = torch.einsum("bhld,hdm->bhlm", q * scale, W)
                kf = torch.einsum("bhld,hdm->bhlm", k, W)
                qf = F.elu(qf) + 1.0
                kf = F.elu(kf) + 1.0

                kv = torch.einsum("bhlm,bhld->bhmd", kf, v)  # (B,H,m,dh)
                z = torch.einsum("bhlm,bhm->bhl", qf, torch.sum(kf, dim=2) + 1e-8)
                out = torch.einsum("bhlm,bhmd->bhld", qf, kv) / (z.unsqueeze(-1) + 1e-8)
                return self.out(self._merge_heads(out))

            if attn_s == "linformer":
                E = self.E
                Fp = self.F
                k_proj = torch.einsum("hml,bhld->bhmd", E, k)
                v_proj = torch.einsum("hml,bhld->bhmd", Fp, v)
                scores = torch.einsum("bhld,bhmd->bhlm", q * scale, k_proj)
                w = torch.softmax(scores, dim=-1)
                out = torch.einsum("bhlm,bhmd->bhld", w, v_proj)
                return self.out(self._merge_heads(out))

            if attn_s == "reformer":
                # Reformer LSH attention (lite): hash tokens into buckets with random projections,
                # then do full attention within sorted chunks (and previous chunk) to approximate.
                if self.R.numel() == 0:
                    raise RuntimeError(
                        "Reformer attention misconfigured (missing projection buffer)"
                    )

                bs = int(reformer_bs)
                out_acc = torch.zeros_like(q)
                n_hash = int(self.R.shape[0])

                for r in range(n_hash):
                    R = self.R[r]  # (H,dh,n_buckets)
                    proj = torch.einsum("bhld,hdm->bhlm", q, R)  # (B,H,L,n_buckets)
                    buckets = proj.argmax(dim=-1)  # (B,H,L)

                    sort_idx = buckets.argsort(dim=-1)  # (B,H,L)
                    gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    q_s = q.gather(dim=2, index=gather_idx)
                    k_s = k.gather(dim=2, index=gather_idx)
                    v_s = v.gather(dim=2, index=gather_idx)

                    L_pad = int(int(math.ceil(float(L) / float(bs))) * bs)
                    pad = int(L_pad - int(L))
                    if pad > 0:
                        q_s = torch.cat([q_s, q_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                        k_s = torch.cat([k_s, k_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                        v_s = torch.cat([v_s, v_s[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)

                    n_chunks = int(L_pad // bs)
                    q_c = q_s.reshape(B, heads, n_chunks, bs, head_dim)
                    k_c = k_s.reshape(B, heads, n_chunks, bs, head_dim)
                    v_c = v_s.reshape(B, heads, n_chunks, bs, head_dim)

                    k_prev = torch.cat([k_c[:, :, :1, :, :], k_c[:, :, :-1, :, :]], dim=2)
                    v_prev = torch.cat([v_c[:, :, :1, :, :], v_c[:, :, :-1, :, :]], dim=2)
                    k_cat = torch.cat([k_prev, k_c], dim=3)  # (B,H,nc,2*bs,dh)
                    v_cat = torch.cat([v_prev, v_c], dim=3)

                    scores = torch.einsum("bhnqd,bhnkd->bhnqk", q_c * scale, k_cat)
                    w = torch.softmax(scores, dim=-1)
                    out_c = torch.einsum("bhnqk,bhnkd->bhnqd", w, v_cat)

                    out_s = out_c.reshape(B, heads, L_pad, head_dim)[:, :, :L, :]
                    inv = sort_idx.argsort(dim=-1)
                    out = out_s.gather(dim=2, index=inv.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                    out_acc = out_acc + out

                out_acc = out_acc / float(max(1, n_hash))
                return self.out(self._merge_heads(out_acc))

            if attn_s == "nystrom":
                m = int(min(int(nystrom_landmarks), L))
                # Landmarks by chunk averaging
                chunk = int(math.ceil(L / m))
                pad = int(m * chunk - L)
                if pad > 0:
                    q_pad = torch.cat([q, q[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                    k_pad = torch.cat([k, k[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                else:
                    q_pad, k_pad = q, k
                q_lm = q_pad.reshape(B, heads, m, chunk, head_dim).mean(dim=3)
                k_lm = k_pad.reshape(B, heads, m, chunk, head_dim).mean(dim=3)

                A = torch.softmax(torch.einsum("bhld,bhmd->bhlm", q * scale, k_lm), dim=-1)
                Bmat = torch.softmax(torch.einsum("bhmd,bhnd->bhmn", q_lm * scale, k_lm), dim=-1)
                C = torch.softmax(torch.einsum("bhmd,bhld->bhml", q_lm * scale, k), dim=-1)
                B_inv = torch.linalg.pinv(Bmat)
                CV = torch.einsum("bhml,bhld->bhmd", C, v)
                out = torch.einsum("bhlm,bhmn,bhnd->bhld", A, B_inv, CV)
                return self.out(self._merge_heads(out))

            if attn_s == "autocorr":
                # AutoCorrelation-style attention (lite): pick top-k delays and aggregate shifted V.
                top_k = int(min(L, auto_k))
                qf = torch.fft.rfft(q, dim=2)
                kf = torch.fft.rfft(k, dim=2)
                corr_f = (qf * torch.conj(kf)).sum(dim=-1)  # (B,H,Lf)
                corr = torch.fft.irfft(corr_f, n=L, dim=2)  # (B,H,L)

                delays = corr.topk(k=top_k, dim=-1).indices  # (B,H,K)
                weights = torch.softmax(corr.gather(dim=-1, index=delays), dim=-1)  # (B,H,K)

                pos = torch.arange(L, device=xb.device).reshape(1, 1, L)
                out = torch.zeros_like(v)
                for i in range(top_k):
                    dly = delays[:, :, i]  # (B,H)
                    idx_t = (pos + dly.unsqueeze(-1)) % int(L)
                    v_shift = v.gather(
                        dim=2, index=idx_t.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
                    )
                    out = out + weights[:, :, i].unsqueeze(-1).unsqueeze(-1) * v_shift
                return self.out(self._merge_heads(out))

            if attn_s == "probsparse":
                # Informer ProbSparse-style attention (lite): compute attention for top-u queries,
                # use mean(V) for the rest.
                scores = torch.einsum("bhld,bhmd->bhlm", q * scale, k)  # (B,H,L,L)
                importance = scores.max(dim=-1).values - scores.mean(dim=-1)  # (B,H,L)
                u = int(min(L, probs_u))
                top_q = importance.topk(k=u, dim=-1).indices  # (B,H,u)

                scores_top = scores.gather(
                    dim=2, index=top_q.unsqueeze(-1).expand(-1, -1, -1, int(L))
                )  # (B,H,u,L)
                w = torch.softmax(scores_top, dim=-1)
                out_top = w @ v  # (B,H,u,dh)

                base = v.mean(dim=2, keepdim=True).expand(-1, -1, int(L), -1).clone()
                out = base.scatter(
                    dim=2,
                    index=top_q.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1]),
                    src=out_top,
                )
                return self.out(self._merge_heads(out))

            scores = torch.einsum("bhld,bhmd->bhlm", q * scale, k)  # (B,H,L,L)
            if attn_s == "local":
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                mask = dist > int(w)
                scores = scores.masked_fill(mask.reshape(1, 1, L, L), float("-inf"))
            if attn_s == "logsparse":
                # LogSparse (LogTrans-style) attention mask (lite):
                # allow local window connections + log-distance (power-of-two) connections.
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()  # (L,L)
                pow2 = (dist > 0) & ((dist & (dist - 1)) == 0)  # powers of two
                allowed = (dist <= int(w)) | pow2
                scores = scores.masked_fill((~allowed).reshape(1, 1, L, L), float("-inf"))
            if attn_s == "longformer":
                # Longformer-style sliding window + global tokens (lite).
                #
                # We treat the future placeholder/query tokens as "global queries" so each horizon token
                # can attend to the full context (and vice versa). This keeps the model usable for
                # direct multi-horizon forecasting while remaining mostly local/sparse.
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()  # (L,L)
                local_allowed = dist <= int(w)

                global_mask = torch.zeros((int(L),), dtype=torch.bool, device=xb.device)
                # Future horizon query tokens are last `h` tokens in the (lags+horizon) sequence.
                if int(h) > 0:
                    global_mask[int(L) - int(h) :] = True
                # Also mark the last observed/context token as global (helps routing information).
                last_ctx = int(L) - int(h) - 1
                if last_ctx >= 0:
                    global_mask[int(last_ctx)] = True

                allowed = (
                    local_allowed | global_mask.reshape(int(L), 1) | global_mask.reshape(1, int(L))
                )
                scores = scores.masked_fill((~allowed).reshape(1, 1, int(L), int(L)), float("-inf"))
            if attn_s == "bigbird":
                # BigBird-style random + local + global (lite).
                w = int(local_window)
                idx = torch.arange(L, device=xb.device)
                dist = (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
                local_allowed = dist <= int(w)

                global_mask = torch.zeros((int(L),), dtype=torch.bool, device=xb.device)
                if int(h) > 0:
                    global_mask[int(L) - int(h) :] = True
                last_ctx = int(L) - int(h) - 1
                if last_ctx >= 0:
                    global_mask[int(last_ctx)] = True

                allowed = (
                    local_allowed | global_mask.reshape(int(L), 1) | global_mask.reshape(1, int(L))
                )
                if self.bigbird_rand.numel() > 0:
                    allowed = allowed | self.bigbird_rand.to(device=xb.device)
                scores = scores.masked_fill((~allowed).reshape(1, 1, int(L), int(L)), float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhlm,bhmd->bhld", w, v)
            return self.out(self._merge_heads(out))

    class _XFormerBlock(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = int(layer_idx)

            self.attn = _MultiheadSelfAttention()
            if norm_s == "rms":
                self.norm1 = _make_rmsnorm(d)
                self.norm2 = _make_rmsnorm(d)
            else:
                self.norm1 = nn.LayerNorm(d)
                self.norm2 = nn.LayerNorm(d)

            self.drop1 = nn.Dropout(p=drop)
            self.drop2 = nn.Dropout(p=drop)

            self.ffn = _SwiGLUFFN() if ffn_s == "swiglu" else _GELUFFN()

            self.gate = None
            if bool(residual_gating):
                self.gate = nn.Linear(d, d)

        def _maybe_gate(self, residual: Any, update: Any) -> Any:
            if self.gate is None:
                return residual + update
            g = torch.sigmoid(self.gate(residual))
            return residual + g * update

        def forward(self, xb: Any) -> Any:
            # Pre-norm
            z = self.attn(self.norm1(xb))
            z = self.drop1(z)
            if drop_path_f > 0.0 and self.training:
                keep = 1.0 - drop_path_f
                shape = (z.shape[0],) + (1,) * (z.ndim - 1)
                mask = (torch.rand(shape, device=z.device) < keep).to(z.dtype)
                z = z * mask / keep
            xb = self._maybe_gate(xb, z)

            z2 = self.ffn(self.norm2(xb))
            z2 = self.drop2(z2)
            if drop_path_f > 0.0 and self.training:
                keep = 1.0 - drop_path_f
                shape = (z2.shape[0],) + (1,) * (z2.ndim - 1)
                mask = (torch.rand(shape, device=z2.device) < keep).to(z2.dtype)
                z2 = z2 * mask / keep
            xb = self._maybe_gate(xb, z2)
            return xb

    class _RevIN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1))

        def forward(self, xb: Any) -> tuple[Any, Any, Any]:
            # xb: (B,L,1)
            mu = xb.mean(dim=1, keepdim=True)
            sig = xb.std(dim=1, keepdim=True)
            sig = torch.clamp(sig, min=1e-6)
            x0 = (xb - mu) / sig
            x0 = x0 * self.gamma + self.beta
            return x0, mu, sig

        def inverse(self, yb: Any, mu: Any, sig: Any) -> Any:
            y0 = (yb - self.beta) / (self.gamma + 1e-6)
            return y0 * sig + mu

    class _XFormerDirect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, d)
            self.horizon_tokens = None
            if horizon_tokens_s == "learned":
                self.horizon_tokens = nn.Parameter(torch.zeros(1, h, 1, dtype=torch.float32))

            # Positional embedding
            if pos_s == "learned":
                self.pos = nn.Parameter(torch.zeros((1, seq_len, d), dtype=torch.float32))
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = None
            elif pos_s == "sincos":
                pe = _positional_encoding_sincos(seq_len, d)
                self.register_buffer(
                    "pos_buf", torch.tensor(pe, dtype=torch.float32).unsqueeze(0), persistent=False
                )
                self.pos = None
                self.time2vec = None
            elif pos_s == "time2vec":
                self.pos = None
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = _Time2Vec()
            else:
                self.pos = None
                self.register_buffer("pos_buf", torch.empty(0), persistent=False)
                self.time2vec = None

            self.blocks = nn.ModuleList([_XFormerBlock(i) for i in range(int(num_layers))])
            self.head = nn.Linear(d, 1)

            self.revin = _RevIN() if bool(revin) else None

        def forward(self, xb: Any) -> Any:
            # xb: (B, seq_len, 1)
            mu = None
            sig = None
            x_in = xb
            if self.revin is not None:
                x_in, mu, sig = self.revin(x_in)

            if self.horizon_tokens is not None:
                x_in = x_in.clone()
                x_in[:, -h:, :] = self.horizon_tokens.expand(x_in.shape[0], -1, -1)

            z = self.embed(x_in)
            if self.pos is not None:
                z = z + self.pos
            elif self.pos_buf.numel() > 0:
                z = z + self.pos_buf
            elif self.time2vec is not None:
                t = torch.linspace(0.0, 1.0, steps=seq_len, device=z.device).reshape(1, seq_len, 1)
                z = z + self.time2vec(t)

            for blk in self.blocks:
                z = blk(z)

            yhat = self.head(z[:, -h:, :]).squeeze(-1)  # (B,h)
            if self.revin is not None and mu is not None and sig is not None:
                # Inverse normalize per-sample. Convert yhat to (B,h,1) then back.
                yhat = self.revin.inverse(yhat.unsqueeze(-1), mu, sig).squeeze(-1)
            return yhat

    model = _XFormerDirect()

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

    feat = x_work[-lag_count:].astype(float, copy=False).reshape(1, lag_count)
    feat_pad = np.concatenate([feat, np.zeros((1, h), dtype=float)], axis=1).reshape(1, seq_len, 1)
    with torch.no_grad():
        feat_t = torch.tensor(feat_pad, dtype=torch.float32, device=torch.device(str(device)))
        yhat_t = model(feat_t).detach().cpu().numpy().reshape(-1)

    yhat = yhat_t.astype(float, copy=False)
    if bool(normalize):
        yhat = yhat * std + mean
    return np.asarray(yhat, dtype=float)
