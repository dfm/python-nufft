# -*- coding: utf-8 -*-

__version__ = "0.0.1-dev"

try:
    __NUFFT_SETUP__
except NameError:
    __NUFFT_SETUP__ = False

if not __NUFFT_SETUP__:
    __all__ = ["nufft1", "nufft3"]
    from .nufft import nufft1, nufft3
