# -*- coding: utf-8 -*-

from __future__ import division, print_function
import unittest
import numpy as np
from nufft import nufft1d1, nufft1d2, nufft1d3


def _error(exact, approx):
    return np.sqrt(np.sum(np.abs(exact - approx) ** 2) / np.sum(np.abs(exact)**2))


class NUFFT1DTestCase(unittest.TestCase):
    """Tests for 1D `nufft.py`.

    The purpose of this script is to test the NUFFT library through
    the python-nufft interface. This seeks to validate the NUFFT
    implementation for the special case where it corresponds to a DFT.
    The script transforms 1-dimensional arrays between the space/time
    and the Fourier domain and compares the output to that of the
    corresponding FFT implementation from NumPy.

    """

    def setUp(self):

        # Parameters
        self.N = 32
        self.eps = 1e-13

        # Coordinates
        self.x = 2 * np.pi * np.arange(self.N) / self.N

        self.c = self.x

        # Frequency points
        self.st = np.arange(self.N)

        # Numpy baseline FFT
        self.f_numpy = np.fft.fft(self.c)
        self.c_numpy = np.fft.ifft(self.f_numpy)

    def test_type1_dft(self):
        """Is the NUFFT type 1 DFT correct?"""
        f_dir1 = np.roll(nufft1d1(self.x,
                                  self.c,
                                  self.N,
                                  iflag=-1,
                                  direct=True),
                         -int(self.N / 2),
                         0) * self.N
        f_nufft1 = np.roll(nufft1d1(self.x,
                                    self.c,
                                    self.N,
                                    iflag=-1,
                                    direct=False),
                           -int(self.N / 2),
                           0) * self.N

        self.assertTrue(_error(self.f_numpy, f_dir1) < self.eps,
                        "NUFFT direct DFT (1) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.f_numpy, f_nufft1) < self.eps,
                        "NUFFT direct DFT (1) vs. NumPy IFFT: error too large")

    def test_type1_idft(self):
        """Is the NUFFT type 1 IDFT correct?"""
        c_dir = np.roll(nufft1d1(self.x,
                                 self.f_numpy,
                                 self.N,
                                 iflag=1,
                                 direct=True),
                        -int(self.N / 2),
                        0)
        c_nufft = np.roll(nufft1d1(self.x,
                                   self.f_numpy,
                                   self.N,
                                   iflag=1,
                                   direct=False),
                          -int(self.N / 2),
                          0)

        self.assertTrue(_error(self.c_numpy, c_dir) < self.eps,
                        "NUFFT direct IDFT (1) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy, c_nufft) < self.eps,
                        "NUFFT direct IDFT (1) vs. NumPy IFFT: error too large")

    def test_type2_dft(self):
        """Is the NUFFT type 2 DFT correct?"""
        f_dir2 = nufft1d2(self.x,
                          np.roll(self.c,
                                  -int(self.N / 2),
                                  0),
                          iflag=-1,
                          direct=True)
        f_nufft2 = nufft1d2(self.x,
                            np.roll(self.c,
                                    -int(self.N / 2),
                                    0),
                            iflag=-1,
                            direct=False)

        self.assertTrue(_error(self.f_numpy, f_dir2) < self.eps,
                        "NUFFT direct DFT (2) vs. NumPy FFT: error too large")
        self.assertTrue(_error(self.f_numpy, f_nufft2) < self.eps,
                        "NUFFT FFT (2) vs. NumPy FFT: error too large")

    def test_type2_idft(self):
        """Is the NUFFT type 2 IDFT correct?"""
        c_dir2 = nufft1d2(self.x,
                          np.roll(self.f_numpy,
                                  -int(self.N / 2),
                                  0),
                          iflag=1,
                          direct=True) / self.N
        c_nufft2 = nufft1d2(self.x,
                            np.roll(self.f_numpy,
                                    -int(self.N / 2),
                                    0),
                            iflag=1,
                            direct=False) / self.N

        self.assertTrue(_error(self.c_numpy, c_dir2) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy, c_nufft2) < self.eps,
                        "NUFFT IFFT (2) vs. NumPy IFFT: error too large")

    def test_type3_dft(self):
        """Is the NUFFT type 3 DFT correct?"""
        f_dir3 = nufft1d3(self.x,
                          self.c,
                          self.st,
                          iflag=-1,
                          direct=True) * self.N
        f_nufft3 = nufft1d3(self.x,
                            self.c,
                            self.st,
                            iflag=-1,
                            direct=False) * self.N

        self.assertTrue(_error(self.f_numpy, f_dir3) < self.eps,
                        "NUFFT direct DFT (3) vs. NumPy FFT: error too large")
        self.assertTrue(_error(self.f_numpy, f_nufft3) < self.eps,
                        "NUFFT FFT (3) vs. NumPy FFT: error too large")

    def test_type3_idft(self):
        """Is the NUFFT type 3 IDFT correct?"""
        c_dir3 = nufft1d3(self.x,
                          self.f_numpy,
                          self.st,
                          iflag=1,
                          direct=True)
        c_nufft3 = nufft1d3(self.x,
                            self.f_numpy,
                            self.st,
                            iflag=1,
                            direct=False)

        self.assertTrue(_error(self.c_numpy, c_dir3) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy, c_nufft3) < self.eps,
                        "NUFFT IFFT (2) error vs. NumPy IFFT: error too large")


if __name__ == '__main__':
    unittest.main()
