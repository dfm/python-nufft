# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_type_3"]

import numpy as np
from .nufft import nufft3


def _type_3(eps=1e-10):
    ms = 90
    nj = 128
    k1 = np.arange(-0.5 * nj, 0.5 * nj)
    j = k1 + 0.5 * nj + 1
    x = np.pi * np.cos(-np.pi * j / nj)
    y = np.empty_like(x, dtype=np.complex128)
    y.real = np.sin(np.pi * j / nj)
    y.imag = np.cos(np.pi * j / nj)
    f = 48 * np.cos((np.arange(ms) + 1) * np.pi / ms)
    p1 = nufft3(x, y, f, eps=eps)
    p2 = nufft3(x, y, f, direct=True)

    err = np.sqrt(np.sum((p1 - p2) ** 2) / np.sum(p1**2))
    assert err < eps


def test_type_3():
    for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
        _type_3(eps)
