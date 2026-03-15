import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import math
from functools import partial
from sympy import Poly, legendre, Symbol, chebyshevt
from scipy.special import eval_legendre


def legendre_der(k, x):
    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)

    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(
                    np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol('x')
        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / k_use / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2


def get_filter(base, k):
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')

    x = Symbol('x')
    h0 = np.zeros((k, k))
    h1 = np.zeros((k, k))
    g0 = np.zeros((k, k))
    g1 = np.zeros((k, k))
    phi0 = np.zeros((k, k))
    phi1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendre_der(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                h0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                g0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                h1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                g1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        phi0 = np.eye(k)
        phi1 = np.eye(k)

    elif base == 'chebyshev':
        x = Symbol('x')
        k_use = 2 * k
        roots = Poly(chebyshevt(k_use, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / k_use / 2

        for ki in range(k):
            for kpi in range(k):
                h0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                g0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                h1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                g1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

                phi0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                phi1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2

        phi0[np.abs(phi0) < 1e-8] = 0
        phi1[np.abs(phi1) < 1e-8] = 0

    h0[np.abs(h0) < 1e-8] = 0
    h1[np.abs(h1) < 1e-8] = 0
    g0[np.abs(g0) < 1e-8] = 0
    g1[np.abs(g1) < 1e-8] = 0

    return h0, h1, g0, g1, phi0, phi1


class MultiWaveletTransform(nn.Module):
    """
    1D multiwavelet block.
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128,
                 n_cz=1, levels=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('base', base)
        self.k = k
        self.c = c
        self.L = levels
        self.n_cz = n_cz
        self.Lk0 = nn.Linear(ich, c * k)
        self.Lk1 = nn.Linear(c * k, ich)
        self.ich = ich
        self.mwt_cz = nn.ModuleList(MWT_CZ1d(k, alpha, levels, c, base) for _ in range(n_cz))

    def forward(self, queries, keys, values, attn_mask):
        batch_size, seq_len, _, _ = queries.shape
        _, source_steps, _, value_dim = values.shape
        if seq_len > source_steps:
            zeros = torch.zeros_like(queries[:, :(seq_len - source_steps), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :seq_len, :, :]
            keys = keys[:, :seq_len, :, :]
        values = values.view(batch_size, seq_len, -1)

        values_proj = self.Lk0(values).view(batch_size, seq_len, self.c, -1)
        for index in range(self.n_cz):
            values_proj = self.mwt_cz[index](values_proj)
            if index < self.n_cz - 1:
                values_proj = F.relu(values_proj)

        values_proj = self.Lk1(values_proj.view(batch_size, seq_len, -1))
        values_proj = values_proj.view(batch_size, seq_len, -1, value_dim)
        return (values_proj.contiguous(), None)


class MultiWaveletCross(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 levels=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(MultiWaveletCross, self).__init__()
        print('base', base)

        self.c = c
        self.k = k
        self.L = levels
        h0, h1, g0, g1, phi0, phi1 = get_filter(base, k)
        h0_r = h0 @ phi0
        g0_r = g0 @ phi0
        h1_r = h1 @ phi1
        g1_r = g1 @ phi1

        h0_r[np.abs(h0_r) < 1e-8] = 0
        h1_r[np.abs(h1_r) < 1e-8] = 0
        g0_r[np.abs(g0_r) < 1e-8] = 0
        g1_r[np.abs(g1_r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = FourierCrossAttentionW(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((h0.T, h1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((g0.T, g1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((h0_r, g0_r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((h1_r, g1_r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        batch_size, query_steps, _, _ = q.shape  # (B, N, H, E)
        _, key_steps, _, _ = k.shape  # (B, S, H, E)

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if query_steps > key_steps:
            zeros = torch.zeros_like(q[:, :(query_steps - key_steps), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :query_steps, :, :]
            k = k[:, :query_steps, :, :]

        ns = math.floor(np.log2(query_steps))
        nl = pow(2, math.ceil(np.log2(query_steps)))
        extra_q = q[:, 0:nl - query_steps, :, :]
        extra_k = k[:, 0:nl - query_steps, :, :]
        extra_v = v[:, 0:nl - query_steps, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        detail_query_pairs = torch.jit.annotate(List[Tuple[Tensor]], [])
        detail_key_pairs = torch.jit.annotate(List[Tuple[Tensor]], [])
        detail_value_pairs = torch.jit.annotate(List[Tuple[Tensor]], [])

        smooth_queries = torch.jit.annotate(List[Tensor], [])
        smooth_keys = torch.jit.annotate(List[Tensor], [])
        smooth_values = torch.jit.annotate(List[Tensor], [])

        detail_outputs = torch.jit.annotate(List[Tensor], [])
        smooth_outputs = torch.jit.annotate(List[Tensor], [])

        # decompose
        for _ in range(ns - self.L):
            d, q = self.wavelet_transform(q)
            detail_query_pairs += [tuple([d, q])]
            smooth_queries += [d]
        for _ in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            detail_key_pairs += [tuple([d, k])]
            smooth_keys += [d]
        for _ in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            detail_value_pairs += [tuple([d, v])]
            smooth_values += [d]
        for i in range(ns - self.L):
            dk, sk = detail_key_pairs[i], smooth_keys[i]
            dq, sq = detail_query_pairs[i], smooth_queries[i]
            dv, sv = detail_value_pairs[i], smooth_values[i]
            detail_outputs += [
                self.attn1(dq[0], dk[0], dv[0], mask)[0]
                + self.attn2(dq[1], dk[1], dv[1], mask)[0]
            ]
            smooth_outputs += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + smooth_outputs[i]
            v = torch.cat((v, detail_outputs[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :query_steps, :, :].contiguous().view(batch_size, query_steps, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        batch_size, num_steps, c, input_channels = x.shape  # (B, N, c, k)
        assert input_channels == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(batch_size, num_steps * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class FourierCrossAttentionW(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('corss fourier correlation used!')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        batch_size, seq_len, embed_dim, num_heads = q.shape

        xq = q.permute(0, 3, 2, 1)
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(seq_len // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(
            batch_size,
            num_heads,
            embed_dim,
            len(self.index_q),
            device=xq.device,
            dtype=torch.cfloat,
        )
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(
            batch_size,
            num_heads,
            embed_dim,
            len(self.index_k_v),
            device=xq.device,
            dtype=torch.cfloat,
        )
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (self.compl_mul1d("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = self.compl_mul1d("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(
            batch_size,
            num_heads,
            embed_dim,
            seq_len // 2 + 1,
            device=xq.device,
            dtype=torch.cfloat,
        )
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        return (out, None)


class SparseKernelFt1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super().__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.float))
        self.weights1.requires_grad = True
        self.weights2.requires_grad = True
        self.k = k

    def compl_mul1d(self, order, x, weights):
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        if x_flag or w_flag:
            return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, x):
        batch_size, num_steps, c, k = x.shape  # (B, N, c, k)

        x = x.view(batch_size, num_steps, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, num_steps // 2 + 1)
        out_ft = torch.zeros(batch_size, c * k, num_steps // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = self.compl_mul1d("bix,iox->box", x_fft[:, :, :l],
                                            torch.complex(self.weights1, self.weights2)[:, :, :l])
        x = torch.fft.irfft(out_ft, n=num_steps)
        x = x.permute(0, 2, 1).view(batch_size, num_steps, c, k)
        return x


# ##
class MwtCz1d(nn.Module):
    def __init__(self,
                 k=3, alpha=64,
                 levels=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super().__init__()

        self.k = k
        self.L = levels
        h0, h1, g0, g1, phi0, phi1 = get_filter(base, k)
        h0_r = h0 @ phi0
        g0_r = g0 @ phi0
        h1_r = h1 @ phi1
        g1_r = g1 @ phi1

        h0_r[np.abs(h0_r) < 1e-8] = 0
        h1_r[np.abs(h1_r) < 1e-8] = 0
        g0_r[np.abs(g0_r) < 1e-8] = 0
        g1_r[np.abs(g1_r) < 1e-8] = 0
        self.max_item = 3

        self.A = SparseKernelFt1d(k, alpha, c)
        self.B = SparseKernelFt1d(k, alpha, c)
        self.C = SparseKernelFt1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((h0.T, h1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((g0.T, g1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((h0_r, g0_r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((h1_r, g1_r), axis=0)))

    def forward(self, x):
        _, num_steps, _, _ = x.shape  # (B, N, k)
        ns = math.floor(np.log2(num_steps))
        nl = pow(2, math.ceil(np.log2(num_steps)))
        extra_x = x[:, 0:nl - num_steps, :, :]
        x = torch.cat([x, extra_x], 1)
        update_details = torch.jit.annotate(List[Tensor], [])
        update_smooths = torch.jit.annotate(List[Tensor], [])
        for _ in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            update_details += [self.A(d) + self.B(x)]
            update_smooths += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + update_smooths[i]
            x = torch.cat((x, update_details[i]), -1)
            x = self.evenOdd(x)
        x = x[:, :num_steps, :, :]

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        batch_size, num_steps, c, input_channels = x.shape  # (B, N, c, k)
        assert input_channels == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(batch_size, num_steps * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


sparseKernelFT1d = SparseKernelFt1d
MWT_CZ1d = MwtCz1d
