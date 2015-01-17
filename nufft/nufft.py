# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["nufft"]

import numpy as np
from ._nufft import (
    wrap_dirft1d1,
    wrap_nufft1d3, wrap_dirft1d3
)


def nufft1(x, y, mk, eps=1e-10, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    if len(x) != len(y):
        raise ValueError("Dimension mismatch")

    # Rescale the x values into [-pi, pi]
    mn, mx = np.min(x), np.max(x)
    rng = mx - mn
    x = (2 * (x - mn) / rng - 1) * np.pi

    # Run the Fortran code.
    if direct:
        p = wrap_dirft1d1(x, y, iflag, mk)
    else:
        assert 0
        # p, flag = wrap_nufft1d1(x, y, iflag, eps, f)
        # # Check the output and return.
        # if flag:
        #     raise RuntimeError("nufft1d3 failed with code {0}".format(flag))

    f = rng * np.linspace(-1, 1, )

    return f, p


def nufft3(x, y, f, eps=1e-10, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    if len(x) != len(y):
        raise ValueError("Dimension mismatch")

    # Make sure that the frequencies are of the right type.
    f = np.ascontiguousarray(f, dtype=np.float64)

    # Run the Fortran code.
    if direct:
        p = wrap_dirft1d3(x, y, iflag, f)
    else:
        p, flag = wrap_nufft1d3(x, y, iflag, eps, f)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft1d3 failed with code {0}".format(flag))

    return p
