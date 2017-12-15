# -*- coding: utf-8 -*-

__version__ = "0.0.1-dev"

try:
    __NUFFT_SETUP__
except NameError:
    __NUFFT_SETUP__ = False

if not __NUFFT_SETUP__:
    __all__ = ["nufft1d1freqs",
               "nufft1d1",
               "nufft1d2",
               "nufft1d3",
               "nufft2d1",
               "nufft2d2",
               "nufft2d3",
               "nufft3d1",
               "nufft3d2",
               "nufft3d3"]
    from .nufft import nufft1d1freqs, nufft1d1, nufft1d2, nufft1d3, nufft2d1, nufft2d2, nufft2d3, nufft3d1, nufft3d2, nufft3d3
