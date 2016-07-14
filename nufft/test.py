# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_type_1", "test_type_2", "test_type_3", "test_1_and_3"]

import numpy as np
from .nufft import nufft1d1freqs, nufft1d1, nufft1d2, nufft1d3


def _get_data():
    ms = 90
    nj = 128
    k1 = np.arange(-0.5 * nj, 0.5 * nj)
    j = k1 + 0.5 * nj + 1
    x = np.pi * np.cos(-np.pi * j / nj)
    y = np.empty_like(x, dtype=np.complex128)
    y.real = np.sin(np.pi * j / nj)
    y.imag = np.cos(np.pi * j / nj)
    f = 48 * np.cos((np.arange(ms) + 1) * np.pi / ms)
    return x, y, f

def _get_data_2d():
    ms = 20
    mt = 20
    nj = 128
    k1 = np.arange(-0.5 * nj, 0.5 * nj)
    j = k1 + 0.5 * nj + 1
    x = np.pi * np.cos(-np.pi * j / nj)
    y = np.pi * np.sin(-np.pi * j / nj)
    z = np.empty_like(x, dtype=np.complex128)
    z.real = np.sin(np.pi * j / nj)
    z.imag = np.cos(np.pi * j / nj)
    ms_val, mt_val = np.meshgrid(np.arange(ms)/ms, np.arange(mt)/mt, sparse=True)
    f = 48 * np.cos((ms_val + mt_val + 1) * np.pi)
    return x, y, f


def _get_data_roundtrip():
    ms = 512
    nj = 200
    x = np.sort(np.random.choice(np.linspace(-np.pi, np.pi, ms, endpoint=False), nj, replace=False))
    y = np.random.randn(nj)
    f = np.empty(ms)
    return x, y, f


def _type_1(eps=1e-10):
    x, y, f = _get_data()
    p2 = nufft1(x, y, len(f), direct=True)
    p1 = nufft1(x, y, len(f), eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps
    assert len(nufft1freqs(len(f))) == len(p1), "even"

    p2 = nufft1(x, y, len(f) + 1, direct=True)
    p1 = nufft1(x, y, len(f) + 1, eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps
    assert len(nufft1freqs(len(f) + 1)) == len(p1), "odd"


def test_type_1():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_1(eps)


def _type_2(eps=1e-10):
    x, y, f = _get_data()
    y2 = nufft2(x, f, direct=True)
    y1 = nufft2(x, f, eps=eps)
    err = np.sqrt(np.sum(np.abs(y1 - y2) ** 2) / np.sum(np.abs(y1)**2))
    assert err < eps


def test_type_2():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_2(eps)


def _type_1_2_roundtrip(eps=1e-10):
    x, y1, f = _get_data_roundtrip()
    p = nufft1(x, y1, len(f), iflag=-1, eps=eps)
    y2 = len(x) / len(f) * nufft2(x, p, iflag=1, direct=True)
    err = np.sqrt(np.sum(np.abs(y1 - y2) ** 2) / np.sum(np.abs(y1)**2))
    assert err < eps


def test_type_1_2_roundtrip():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_1_2_roundtrip(eps)


def _type_3(eps=1e-10):
    x, y, f = _get_data()
    p2 = nufft3(x, y, f, direct=True)
    p1 = nufft3(x, y, f, eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps


def test_type_3():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_3(eps)


def _type_1_and_3(eps=1e-10):
    x, y, f = _get_data()

    f = nufft1freqs(len(f))
    p2 = nufft3(x, y, f, eps=eps)
    p1 = nufft1(x, y, len(f), eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps, "even"

    f = nufft1freqs(len(f) + 1)
    p2 = nufft3(x, y, f, eps=eps)
    p1 = nufft1(x, y, len(f), eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps, "odd"

    df = 0.5 * (f[1] - f[0])
    p1 = nufft1(x, y, len(f), eps=eps, df=df)
    f = nufft1freqs(len(f), df=df)
    p2 = nufft3(x, y, f, eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps, "even"


def test_1_and_3():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_1_and_3(eps)
