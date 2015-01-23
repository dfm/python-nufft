# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_type_1", "test_type_3"]

import numpy as np
from .nufft import nufft1freqs, nufft1, nufft3


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


def _type_1(eps=1e-10):
    x, y, f = _get_data()
    p1 = nufft1(x, y, len(f), eps=eps)
    p2 = nufft1(x, y, len(f), direct=True)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps
    assert len(nufft1freqs(len(f))) == len(p1)

    p1 = nufft1(x, y, len(f) + 1, eps=eps)
    p2 = nufft1(x, y, len(f) + 1, direct=True)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps
    assert len(nufft1freqs(len(f) + 1)) == len(p1)


def test_type_1():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_1(eps)


def _type_3(eps=1e-10):
    x, y, f = _get_data()
    p1 = nufft3(x, y, f, eps=eps)
    p2 = nufft3(x, y, f, direct=True)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps


def test_type_3():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_3(eps)


def _type_1_and_3(eps=1e-10):
    x, y, f = _get_data()

    p1 = nufft1(x, y, len(f), eps=eps)
    f = nufft1freqs(len(f))
    print(len(f), len(p1))
    p2 = nufft3(x, y, f, eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps

    p1 = nufft1(x, y, len(f) + 1, eps=eps)
    f = nufft1freqs(len(f) + 1)
    p2 = nufft3(x, y, f, eps=eps)
    err = np.sqrt(np.sum(np.abs(p1 - p2) ** 2) / np.sum(np.abs(p1)**2))
    assert err < eps


def test_1_and_3():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_1_and_3(eps)
