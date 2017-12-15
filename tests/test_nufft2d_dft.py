# -*- coding: utf-8 -*-

from __future__ import division, print_function
import unittest
import numpy as np
from nufft import nufft2d1, nufft2d2, nufft2d3


class NUFFT2DTestCase(unittest.TestCase):
    """Tests for 2D `nufft.py`.

    The purpose of this script is to test the NUFFT library through
    the python-nufft interface. This seeks to validate the NUFFT
    implementation for the special case where it corresponds to a DFT.
    The script transforms 2-dimensional arrays between the space/time
    and the Fourier domain and compares the output to that of the
    corresponding FFT implementation from NumPy.

    """

    def setUp(self):

        # Parameters
        self.num_samples = 32 * np.ones(2, dtype=int)
        self.eps = 1e-13

        # Coordinates
        self.x = 2 * np.pi * \
            np.arange(self.num_samples[0]) / self.num_samples[0]
        self.y = 2 * np.pi * \
            np.arange(self.num_samples[1]) / self.num_samples[1]
        self.xy_grid = np.meshgrid(self.x, self.y)

        self.c = self.xy_grid[0] + self.xy_grid[1]

        # Frequency points
        self.st_grid = np.meshgrid(np.arange(self.num_samples[0]),
                                   np.arange(self.num_samples[1]))

        # Numpy baseline FFT
        self.f_numpy = np.fft.fft2(self.c)
        self.c_numpy = np.fft.ifft2(self.f_numpy)

    def test_type1_dft(self):
        """Is the NUFFT type 1 DFT correct?"""
        f_dir1 = np.roll(np.roll(nufft2d1(self.xy_grid[0].reshape(-1),
                                          self.xy_grid[1].reshape(-1),
                                          self.c.reshape(-1),
                                          self.num_samples[0],
                                          self.num_samples[1],
                                          iflag=-1,
                                          direct=True),
                                 -int(self.num_samples[0] / 2),
                                 0),
                         -int(self.num_samples[1] / 2),
                         1) * self.num_samples.prod()
        f_nufft1 = np.roll(np.roll(nufft2d1(self.xy_grid[0].reshape(-1),
                                            self.xy_grid[1].reshape(-1),
                                            self.c.reshape(-1),
                                            self.num_samples[0],
                                            self.num_samples[1],
                                            iflag=-1,
                                            direct=False),
                                   -int(self.num_samples[0] / 2),
                                   0),
                           -int(self.num_samples[1] / 2),
                           1) * self.num_samples.prod()

        self.assertTrue(np.sqrt(np.sum(np.abs(f_dir1 - self.f_numpy).reshape(-1) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct DFT (1) vs. NumPy FFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(f_nufft1 - self.f_numpy).reshape(-1) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT FFT (1) vs. NumPy FFT: error too large")

    def test_type1_idft(self):
        """Is the NUFFT type 1 IDFT correct?"""

        ### NUFFT type 1 IDFT transforms ###
        c_dir = np.roll(np.roll(nufft2d1(self.xy_grid[0].reshape(-1),
                                         self.xy_grid[1].reshape(-1),
                                         self.f_numpy.reshape(-1),
                                         self.num_samples[0], self.num_samples[1],
                                         iflag=1, direct=True),
                                -int(self.num_samples[0] / 2), 0),
                        -int(self.num_samples[1] / 2), 1)
        c_nufft = np.roll(np.roll(nufft2d1(self.xy_grid[0].reshape(-1),
                                           self.xy_grid[1].reshape(-1),
                                           self.f_numpy.reshape(-1),
                                           self.num_samples[0],
                                           self.num_samples[1], iflag=1,
                                           direct=False),
                                  -int(self.num_samples[0] / 2), 0),
                          -int(self.num_samples[1] / 2), 1)

        self.assertTrue(np.sqrt(np.sum(np.abs(c_dir - self.c_numpy).reshape(-1) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct IDFT (1) vs. NumPy IFFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(c_nufft - self.c_numpy).reshape(-1) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT IFFT (1) vs. NumPy IFFT: error too large")

    def test_type2_dft(self):
        """Is the NUFFT type 2 DFT correct?"""
        f_dir2 = nufft2d2(self.xy_grid[0].reshape(-1),
                          self.xy_grid[1].reshape(-1),
                          np.roll(np.roll(self.c,
                                          -int(self.num_samples[0] / 2),
                                          0),
                                  -int(self.num_samples[1] / 2),
                                  1),
                          iflag=-1,
                          direct=True)
        f_nufft2 = nufft2d2(self.xy_grid[0].reshape(-1),
                            self.xy_grid[1].reshape(-1),
                            np.roll(np.roll(self.c,
                                            -int(self.num_samples[0] / 2),
                                            0),
                                    -int(self.num_samples[1] / 2),
                                    1),
                            iflag=-1,
                            direct=False)

        self.assertTrue(np.sqrt(np.sum(np.abs(f_dir2 - self.f_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct DFT (2) vs. NumPy FFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(f_nufft2 - self.f_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT FFT (2) vs. NumPy FFT: error too large")

    def test_type2_idft(self):
        """Is the NUFFT type 2 IDFT correct?"""
        c_dir2 = nufft2d2(self.xy_grid[0].reshape(-1),
                          self.xy_grid[1].reshape(-1),
                          np.roll(np.roll(self.f_numpy,
                                          -int(self.num_samples[0] / 2),
                                          0),
                                  -int(self.num_samples[1] / 2),
                                  1),
                          iflag=1,
                          direct=True) / self.num_samples.prod()
        c_nufft2 = nufft2d2(self.xy_grid[0].reshape(-1),
                            self.xy_grid[1].reshape(-1),
                            np.roll(np.roll(self.f_numpy,
                                            -int(self.num_samples[0] / 2),
                                            0),
                                    -int(self.num_samples[1] / 2),
                                    1),
                            iflag=1,
                            direct=False) / self.num_samples.prod()

        self.assertTrue(np.sqrt(np.sum(np.abs(c_dir2 - self.c_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(c_nufft2 - self.c_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT IFFT (2) vs. NumPy IFFT: error too large")

    def test_type3_dft(self):
        """Is the NUFFT type 3 DFT correct?"""
        f_dir3 = nufft2d3(self.xy_grid[0].reshape(-1),
                          self.xy_grid[1].reshape(-1),
                          self.c.reshape(-1),
                          self.st_grid[0].reshape(-1),
                          self.st_grid[1].reshape(-1),
                          iflag=-1,
                          direct=True) * self.num_samples.prod()
        f_nufft3 = nufft2d3(self.xy_grid[0].reshape(-1),
                            self.xy_grid[1].reshape(-1),
                            self.c.reshape(-1),
                            self.st_grid[0].reshape(-1),
                            self.st_grid[1].reshape(-1),
                            iflag=-1,
                            direct=False) * self.num_samples.prod()

        self.assertTrue(np.sqrt(np.sum(np.abs(f_dir3 - self.f_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct DFT (3) vs. NumPy FFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(f_nufft3 - self.f_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.f_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT FFT (3) vs. NumPy FFT: error too large")

    def test_type3_idft(self):
        """Is the NUFFT type 3 IDFT correct?"""
        c_dir3 = nufft2d3(self.xy_grid[0].reshape(-1),
                          self.xy_grid[1].reshape(-1),
                          self.f_numpy.reshape(-1),
                          self.st_grid[0].reshape(-1),
                          self.st_grid[1].reshape(-1),
                          iflag=1,
                          direct=True)
        c_nufft3 = nufft2d3(self.xy_grid[0].reshape(-1),
                            self.xy_grid[1].reshape(-1),
                            self.f_numpy.reshape(-1),
                            self.st_grid[0].reshape(-1),
                            self.st_grid[1].reshape(-1),
                            iflag=1,
                            direct=False)

        self.assertTrue(np.sqrt(np.sum(np.abs(c_dir3 - self.c_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(np.sqrt(np.sum(np.abs(c_nufft3 - self.c_numpy.reshape(-1)) ** 2) /
                                np.sum(np.abs(self.c_numpy).reshape(-1)**2)) < self.eps,
                        "NUFFT IFFT (2) error vs. NumPy IFFT: error too large")


if __name__ == '__main__':
    unittest.main()
