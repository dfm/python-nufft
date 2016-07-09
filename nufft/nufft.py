# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["nufft1d1freqs", "nufft1d1", "nufft1d2", "nufft1d3",
           "nufft2d1", "nufft2d2"]

import numpy as np
from ._nufft import (
    dirft1d1, nufft1d1f90,
    dirft1d2, nufft1d2f90,
    dirft1d3, nufft1d3f90,
    dirft2d1, nufft2d1f90,
    dirft2d2, nufft2d2f90,
)


def nufft1d1freqs(ms, df=1.0):
    return df * (np.arange(-ms // 2, ms // 2) + ms % 2)


def nufft1d1(x, y, ms, df=1.0, eps=1e-15, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    if len(x) != len(y):
        raise ValueError("Dimension mismatch")

    # Run the Fortran code.
    if direct:
        p = dirft1d1(x * df, y, iflag, ms)
    else:
        p, flag = nufft1d1f90(x * df, y, iflag, eps, ms)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft1d1 failed with code {0}".format(flag))
    return p


def nufft1d2(x, p, df=1.0, eps=1e-15, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.complex128)

    # Run the Fortran code.
    if direct:
        y = dirft1d2(x * df, iflag, p)
    else:
        y, flag = nufft1d2f90(x * df, iflag, eps, p)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft1d2 failed with code {0}".format(flag))
    return y


def nufft1d3(x, y, f, eps=1e-15, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.complex128)
    if len(x) != len(y):
        raise ValueError("Dimension mismatch")

    # Make sure that the frequencies are of the right type.
    f = np.ascontiguousarray(f, dtype=np.float64)

    # Run the Fortran code.
    if direct:
        p = dirft1d3(x, y, iflag, f)
    else:
        p, flag = nufft1d3f90(x, y, iflag, eps, f)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft1d3 failed with code {0}".format(flag))
    return p / len(x)


def nufft2d1(x, y, z, ms, mt, df=1.0, eps=1e-15, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.complex128)
    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("Dimension mismatch")

    # Run the Fortran code.
    if direct:
        p = dirft2d1(x * df, y * df, z, iflag, ms, mt)
    else:
        p, flag = nufft2d1f90(x * df, y * df, z, iflag, eps, ms, mt)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft2d1 failed with code {0}".format(flag))
    return p


def nufft2d2(x, y, p, df=1.0, eps=1e-15, iflag=1, direct=False):
    # Make sure that the data are properly formatted.
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.complex128)

    # Run the Fortran code.
    if direct:
        z = dirft2d2(x * df, y * df, iflag, p)
    else:
        z, flag = nufft2d2f90(x * df, y * df, iflag, eps, p)
        # Check the output and return.
        if flag:
            raise RuntimeError("nufft2d2 failed with code {0}".format(flag))
    return z
