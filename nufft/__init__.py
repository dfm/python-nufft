# -*- coding: utf-8 -*-

__version__ = "0.0.1-dev"

try:
    __NUFFT_SETUP__
except NameError:
    __NUFFT_SETUP__ = False

if not __NUFFT_SETUP__:
    __all__ = ["nufft1freqs", "nufft1", "nufft2", "nufft3"]
    from .nufft import nufft1freqs, nufft1, nufft2, nufft3
