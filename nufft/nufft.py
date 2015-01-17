# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["nufft"]

import numpy as np
from ._nufft import wrap_nufft1d3, wrap_dirft1d3


def nufft(x, y, f, eps=1e-10, iflag=1, direct=False):
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
