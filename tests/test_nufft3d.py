# -*- coding: utf-8 -*-

from __future__ import division, print_function
import unittest
import numpy as np
from nufft import nufft3d1, nufft3d2, nufft3d3


def _error(exact, approx):
    return np.sqrt(np.sum(np.abs(exact - approx) ** 2) / np.sum(np.abs(exact)**2))


class NUFFT3DTestCase(unittest.TestCase):
    """Tests for 3D `nufft.py`.

    The purpose of this script is to test the NUFFT library through
    the python-nufft interface. In addition to the usual interface
    checks, this seeks to validate the NUFFT implementation for the
    special case where it corresponds to a DFT.  The script transforms
    3-dimensional arrays between the space/time and the Fourier domain
    and compares the output to that of the corresponding FFT
    implementation from NumPy.

    """

    def setUp(self):

        # Parameters
        self.N = 16 * np.ones(3, dtype=int)
        self.eps = 1e-13

        # Coordinates
        x = [2 * np.pi * np.arange(self.N[0]) / self.N[0],
             2 * np.pi * np.arange(self.N[1]) / self.N[1],
             2 * np.pi * np.arange(self.N[2]) / self.N[2]]
        self.X = np.meshgrid(x[0], x[1], x[2])

        self.c = self.X[0] + self.X[1] + self.X[2]

        # Frequency points
        self.st_grid = np.meshgrid(np.arange(self.N[0]),
                                   np.arange(self.N[1]),
                                   np.arange(self.N[2]))

        # Numpy baseline FFT
        self.f_numpy = np.fft.fftn(self.c)
        self.c_numpy = np.fft.ifftn(self.f_numpy)
        self.fr_numpy = np.fft.rfftn(self.c)
        self.cr_numpy = np.fft.irfftn(self.fr_numpy)

    def _type_1_even(self, eps=1e-10):

        p2 = nufft3d1(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.N[0],
                      self.N[1],
                      self.N[2],
                      direct=True)
        p1 = nufft3d1(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.N[0],
                      self.N[1],
                      self.N[2],
                      direct=False)
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 1: Discrepancy between direct and fft function")

    def _type_1_odd(self, eps=1e-10):

        p2 = nufft3d1(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.N[0] + 1,
                      self.N[1] + 1,
                      self.N[2] + 1,
                      direct=True)
        p1 = nufft3d1(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.N[0] + 1,
                      self.N[1] + 1,
                      self.N[2] + 1,
                      eps=eps)
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 1: Discrepancy between direct and fft function")

    def _type_2(self, eps=1e-10):
        c2 = nufft3d2(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      np.roll(np.roll(np.roll(self.c,
                                              -int(self.N[0] / 2),
                                              0),
                                      -int(self.N[1] / 2),
                                      1),
                              -int(self.N[2] / 2),
                              2),
                      direct=True)
        c1 = nufft3d2(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      np.roll(np.roll(np.roll(self.c,
                                              -int(self.N[0] / 2),
                                              0),
                                      -int(self.N[1] / 2),
                                      1),
                              -int(self.N[2] / 2),
                              2),
                      eps=eps)
        self.assertTrue(_error(c1, c2) < eps,
                        "Type 2: Discrepancy between direct and fft function")

    def _type_3(self, eps=1e-10):
        p2 = nufft3d3(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.st_grid[0].reshape(-1),
                      self.st_grid[1].reshape(-1),
                      self.st_grid[2].reshape(-1),
                      direct=True)
        p1 = nufft3d3(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.st_grid[0].reshape(-1),
                      self.st_grid[1].reshape(-1),
                      self.st_grid[2].reshape(-1),
                      eps=eps)
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 3: Discrepancy between direct and fft function")

    def _type_1_2_roundtrip(self, eps=1e-10):
        p = nufft3d1(self.X[0].reshape(-1),
                     self.X[1].reshape(-1),
                     self.X[2].reshape(-1),
                     self.c.reshape(-1),
                     self.N[0],
                     self.N[1],
                     self.N[2],
                     iflag=-1,
                     eps=eps)
        c2 = nufft3d2(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      p,
                      iflag=1,
                      direct=True)
        self.assertTrue(_error(self.c.reshape(-1), c2) < eps,
                        "Type 1 and 2: roundtrip error.")

    def _type_1_and_3(self, eps=1e-10):

        p2 = nufft3d3(self.X[0].reshape(-1),
                      self.X[1].reshape(-1),
                      self.X[2].reshape(-1),
                      self.c.reshape(-1),
                      self.st_grid[0].reshape(-1),
                      self.st_grid[1].reshape(-1),
                      self.st_grid[2].reshape(-1),
                      eps=eps)
        p1 = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                              self.X[1].reshape(-1),
                                              self.X[2].reshape(-1),
                                              self.c.reshape(-1),
                                              self.N[0],
                                              self.N[1],
                                              self.N[2],
                                              eps=eps),
                                     -int(self.N[0] / 2),
                                     0),
                             -int(self.N[1] / 2),
                             1),
                     -int(self.N[2] / 2),
                     2)
        self.assertTrue(_error(p1.reshape(-1), p2) < eps,
                        "Type 1 and 3 and not close")

    def test_type_1_even(self):
        """Is the 3D type 1 with even data correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_even(eps)

    def test_type_1_odd(self):
        """Is the 3D type 1 with odd data correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_odd(eps)

    def test_type_2(self):
        """Is the 3D type 2 correct?"""
        for eps in [1e-6, 1e-10, 1e-12]:
            self._type_2(eps)

    def test_type_3(self):
        """Is the 3D type 3 correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_3(eps)

    def test_type_1_2_roundtrip(self):
        """Is the 3D roundtrip using type 1 and 2 correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_2_roundtrip(eps)

    def test_1_and_3(self):
        """Are the 3D type 1 and 3 similar?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_and_3(eps)

    def test_type1_dft(self):
        """Is the NUFFT type 1 DFT correct?"""
        f_dir1 = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                  self.X[1].reshape(-1),
                                                  self.X[2].reshape(-1),
                                                  self.c.reshape(-1),
                                                  self.N[0],
                                                  self.N[1],
                                                  self.N[2],
                                                  iflag=-1,
                                                  direct=True),
                                         -int(self.N[0] / 2),
                                         0),
                                 -int(self.N[1] / 2),
                                 1),
                         -int(self.N[2] / 2),
                         2) * self.N.prod()
        f_nufft1 = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                    self.X[1].reshape(-1),
                                                    self.X[2].reshape(-1),
                                                    self.c.reshape(-1),
                                                    self.N[0],
                                                    self.N[1],
                                                    self.N[2],
                                                    iflag=-1,
                                                    direct=False),
                                           -int(self.N[0] / 2),
                                           0),
                                   -int(self.N[1] / 2),
                                   1),
                           -int(self.N[2] / 2),
                           2) * self.N.prod()

        self.assertTrue(_error(self.f_numpy.reshape(-1), f_dir1.reshape(-1)) < self.eps,
                        "NUFFT direct DFT (1) vs. NumPy FFT: error too large")
        self.assertTrue(_error(self.f_numpy.reshape(-1), f_nufft1.reshape(-1)) < self.eps,
                        "NUFFT FFT (1) vs. NumPy FFT: error too large")

    def test_type1_idft(self):
        """Is the NUFFT type 1 IDFT correct?"""
        c_dir = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                 self.X[1].reshape(-1),
                                                 self.X[2].reshape(-1),
                                                 self.f_numpy.reshape(-1),
                                                 self.N[0],
                                                 self.N[1],
                                                 self.N[2],
                                                 iflag=1,
                                                 direct=True),
                                        -int(self.N[0] / 2),
                                        0),
                                -int(self.N[1] / 2),
                                1),
                        -int(self.N[2] / 2),
                        2)
        c_nufft = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                   self.X[1].reshape(-1),
                                                   self.X[2].reshape(-1),
                                                   self.f_numpy.reshape(-1),
                                                   self.N[0],
                                                   self.N[1],
                                                   self.N[2],
                                                   iflag=1,
                                                   direct=False),
                                          -int(self.N[0] / 2),
                                          0),
                                  -int(self.N[1] / 2),
                                  1),
                          -int(self.N[2] / 2),
                          2)

        self.assertTrue(_error(self.c_numpy.reshape(-1), c_dir.reshape(-1)) < self.eps,
                        "NUFFT direct IDFT (1) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy.reshape(-1), c_nufft.reshape(-1)) < self.eps,
                        "NUFFT IFFT (1) vs. NumPy IFFT: error too large")

    def test_type1_rdft(self):
        """Is the NUFFT type 1 RDFT correct?"""
        f_dir1 = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                  self.X[1].reshape(-1),
                                                  self.X[2].reshape(-1),
                                                  self.c.reshape(-1),
                                                  self.N[0],
                                                  self.N[1],
                                                  self.N[2],
                                                  iflag=-1,
                                                  direct=True),
                                         -int(self.N[0] / 2),
                                         0),
                                 -int(self.N[1] / 2),
                                 1),
                         -int(self.N[2] / 2),
                         2)[:, :, :int(self.N[2] / 2) + 1] * self.N.prod()
        f_nufft1 = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                    self.X[1].reshape(-1),
                                                    self.X[2].reshape(-1),
                                                    self.c.reshape(-1),
                                                    self.N[0],
                                                    self.N[1],
                                                    self.N[2],
                                                    iflag=-1,
                                                    direct=False),
                                           -int(self.N[0] / 2),
                                           0),
                                   -int(self.N[1] / 2),
                                   1),
                           -int(self.N[2] / 2),
                           2)[:, :, :int(self.N[2] / 2) + 1] * self.N.prod()

        self.assertTrue(_error(self.fr_numpy.reshape(-1), f_dir1.reshape(-1)) < self.eps,
                        "NUFFT direct RDFT (1) vs. NumPy RFFT: error too large")
        self.assertTrue(_error(self.fr_numpy.reshape(-1), f_nufft1.reshape(-1)) < self.eps,
                        "NUFFT RFFT (1) vs. NumPy RFFT: error too large")

    def test_type1_irdft(self):
        """Is the NUFFT type 1 IRDFT correct?"""

        # Trick to make it think it is seeing a full FFT
        f = np.concatenate((self.fr_numpy,
                            np.conj(self.fr_numpy[:, :, -2:0:-1])),
                           axis=2)

        c_dir = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                 self.X[1].reshape(-1),
                                                 self.X[2].reshape(-1),
                                                 f.reshape(-1),
                                                 self.N[0],
                                                 self.N[1],
                                                 self.N[2],
                                                 iflag=1,
                                                 direct=True),
                                        -int(self.N[0] / 2),
                                        0),
                                -int(self.N[1] / 2),
                                1),
                        -int(self.N[2] / 2),
                        2)
        c_nufft = np.roll(np.roll(np.roll(nufft3d1(self.X[0].reshape(-1),
                                                   self.X[1].reshape(-1),
                                                   self.X[2].reshape(-1),
                                                   f.reshape(-1),
                                                   self.N[0],
                                                   self.N[1],
                                                   self.N[2],
                                                   iflag=1,
                                                   direct=False),
                                          -int(self.N[0] / 2),
                                          0),
                                  -int(self.N[1] / 2),
                                  1),
                          -int(self.N[2] / 2),
                          2)

        self.assertTrue(_error(self.cr_numpy.reshape(-1), c_dir.reshape(-1)) < self.eps,
                        "NUFFT direct IRDFT (1) vs. NumPy IRFFT: error too large")
        self.assertTrue(_error(self.cr_numpy.reshape(-1), c_nufft.reshape(-1)) < self.eps,
                        "NUFFT IRFFT (1) vs. NumPy IRFFT: error too large")

    def test_type2_dft(self):
        """Is the NUFFT type 2 DFT correct?"""
        f_dir2 = nufft3d2(self.X[0].reshape(-1),
                          self.X[1].reshape(-1),
                          self.X[2].reshape(-1),
                          np.roll(np.roll(np.roll(self.c,
                                                  -int(self.N[0] / 2),
                                                  0),
                                          -int(self.N[1] / 2),
                                          1),
                                  -int(self.N[2] / 2),
                                  2),
                          iflag=-1,
                          direct=True)
        f_nufft2 = nufft3d2(self.X[0].reshape(-1),
                            self.X[1].reshape(-1),
                            self.X[2].reshape(-1),
                            np.roll(np.roll(np.roll(self.c,
                                                    -int(self.N[0] / 2),
                                                    0),
                                            -int(self.N[1] / 2),
                                            1),
                                    -int(self.N[2] / 2),
                                    2),
                            iflag=-1,
                            direct=False)

        self.assertTrue(_error(self.f_numpy.reshape(-1), f_dir2) < self.eps,
                        "NUFFT direct DFT (2) vs. NumPy FFT: error too large")
        self.assertTrue(_error(self.f_numpy.reshape(-1), f_nufft2) < self.eps,
                        "NUFFT FFT (2) vs. NumPy FFT: error too large")

    def test_type2_idft(self):
        """Is the NUFFT type 2 IDFT correct?"""
        c_dir2 = nufft3d2(self.X[0].reshape(-1),
                          self.X[1].reshape(-1),
                          self.X[2].reshape(-1),
                          np.roll(np.roll(np.roll(self.f_numpy,
                                                  -int(self.N[0] / 2),
                                                  0),
                                          -int(self.N[1] / 2),
                                          1),
                                  -int(self.N[2] / 2),
                                  2),
                          iflag=1,
                          direct=True) / self.N.prod()
        c_nufft2 = nufft3d2(self.X[0].reshape(-1),
                            self.X[1].reshape(-1),
                            self.X[2].reshape(-1),
                            np.roll(np.roll(np.roll(self.f_numpy,
                                                    -int(self.N[0] / 2),
                                                    0),
                                            -int(self.N[1] / 2),
                                            1),
                                    -int(self.N[2] / 2),
                                    2),
                            iflag=1,
                            direct=False) / self.N.prod()

        self.assertTrue(_error(self.c_numpy.reshape(-1), c_dir2) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy.reshape(-1), c_nufft2) < self.eps,
                        "NUFFT IFFT (2) vs. NumPy IFFT: error too large")

    def test_type3_dft(self):
        """Is the NUFFT type 3 DFT correct?"""
        f_dir3 = nufft3d3(self.X[0].reshape(-1),
                          self.X[1].reshape(-1),
                          self.X[2].reshape(-1),
                          self.c.reshape(-1),
                          self.st_grid[0].reshape(-1),
                          self.st_grid[1].reshape(-1),
                          self.st_grid[2].reshape(-1),
                          iflag=-1,
                          direct=True) * self.N.prod()
        f_nufft3 = nufft3d3(self.X[0].reshape(-1),
                            self.X[1].reshape(-1),
                            self.X[2].reshape(-1),
                            self.c.reshape(-1),
                            self.st_grid[0].reshape(-1),
                            self.st_grid[1].reshape(-1),
                            self.st_grid[2].reshape(-1),
                            iflag=-1,
                            direct=False) * self.N.prod()

        self.assertTrue(_error(self.f_numpy.reshape(-1), f_dir3) < self.eps,
                        "NUFFT direct DFT (3) vs. NumPy FFT: error too large")
        self.assertTrue(_error(self.f_numpy.reshape(-1), f_nufft3) < self.eps,
                        "NUFFT FFT (3) vs. NumPy FFT: error too large")

    def test_type3_idft(self):
        """Is the NUFFT type 3 IDFT correct?"""
        c_dir3 = nufft3d3(self.X[0].reshape(-1),
                          self.X[1].reshape(-1),
                          self.X[2].reshape(-1),
                          self.f_numpy.reshape(-1),
                          self.st_grid[0].reshape(-1),
                          self.st_grid[1].reshape(-1),
                          self.st_grid[2].reshape(-1),
                          iflag=1,
                          direct=True)
        c_nufft3 = nufft3d3(self.X[0].reshape(-1),
                            self.X[1].reshape(-1),
                            self.X[2].reshape(-1),
                            self.f_numpy.reshape(-1),
                            self.st_grid[0].reshape(-1),
                            self.st_grid[1].reshape(-1),
                            self.st_grid[2].reshape(-1),
                            iflag=1,
                            direct=False)

        self.assertTrue(_error(self.c_numpy.reshape(-1), c_dir3) < self.eps,
                        "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large")
        self.assertTrue(_error(self.c_numpy.reshape(-1), c_nufft3) < self.eps,
                        "NUFFT IFFT (2) error vs. NumPy IFFT: error too large")


if __name__ == '__main__':
    unittest.main()
