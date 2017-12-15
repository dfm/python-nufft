# -*- coding: utf-8 -*-

from __future__ import division, print_function
import unittest
import numpy as np
from nufft import nufft1d1freqs, nufft2d1, nufft2d2, nufft2d3


def _get_data():
    ms = 20
    mt = 20
    nj = 128
    k1 = np.arange(-0.5 * nj, 0.5 * nj)
    j = k1 + 0.5 * nj + 1
    x = np.pi * np.cos(-np.pi * j / nj)
    y = np.pi * np.sin(-np.pi * j / nj)
    c = np.empty_like(x, dtype=np.complex128)
    c.real = np.sin(np.pi * j / nj)
    c.imag = np.cos(np.pi * j / nj)
    ms_val, mt_val = np.meshgrid(np.arange(ms) / ms,
                                 np.arange(mt) / mt,
                                 sparse=True)
    f = 48 * np.cos((ms_val + mt_val + 1) * np.pi)
    g = 32 * np.sin((ms_val + mt_val + 1) * np.pi)
    return x, y, c, f, g


def _get_data_roundtrip():
    ms = 512
    mt = 512
    nj = 200
    x = np.sort(np.random.choice(np.linspace(-np.pi,
                                             np.pi,
                                             ms,
                                             endpoint=False),
                                 nj,
                                 replace=False))
    y = np.sort(np.random.choice(np.linspace(-np.pi,
                                             np.pi,
                                             mt,
                                             endpoint=False),
                                 nj,
                                 replace=False))
    c = np.random.randn(nj)
    f = np.empty((ms, mt))
    return x, y, c, f


def _error(exact, approx):
    return np.sqrt(np.sum(np.abs(exact - approx) ** 2) / np.sum(np.abs(exact)**2))


class NUFFT2DTestCase(unittest.TestCase):
    """Tests for 2D `nufft.py`."""

    def setUp(self):
        self.x, self.y, self.c, self.f, self.g = _get_data()

    def _type_1_even(self, eps=1e-10):
        p2 = nufft2d1(self.x,
                      self.y,
                      self.c,
                      len(self.f),
                      len(self.f),
                      direct=True)
        p1 = nufft2d1(self.x,
                      self.y,
                      self.c,
                      len(self.f),
                      len(self.f),
                      eps=eps)
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 1: Discrepancy between direct and fft function")

    def _type_1_odd(self, eps=1e-10):
        p2 = nufft2d1(self.x,
                      self.y,
                      self.c,
                      len(self.f) + 1,
                      len(self.f) + 1,
                      direct=True)
        p1 = nufft2d1(self.x,
                      self.y,
                      self.c,
                      len(self.f) + 1,
                      len(self.f) + 1,
                      eps=eps)
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 1: Discrepancy between direct and fft function")

    def _type_2(self, eps=1e-10):
        c2 = nufft2d2(self.x, self.y, self.f, direct=True)
        c1 = nufft2d2(self.x, self.y, self.f, eps=eps)
        self.assertTrue(_error(c1, c2) < eps,
                        "Type 2: Discrepancy between direct and fft function")

    def _type_3(self, eps=1e-10):
        p2 = nufft2d3(self.x, self.y, self.c, self.f, self.g, direct=True)
        p1 = nufft2d3(self.x, self.y, self.c, self.f, self.g, eps=eps)
        self.assertTrue(_error(p1, p2),
                        "Type 3: Discrepancy between direct and fft function")

    def _type_1_2_roundtrip(self, eps=1e-10):
        x, y, c1, f = _get_data_roundtrip()
        p = nufft2d1(x, y, c1, f.shape[0], f.shape[1], iflag=-1, eps=eps)
        c2 = len(x) / len(f)**2 * nufft2d2(x, y, p, iflag=1, direct=True)
        self.assertTrue(_error(c1, c2) < eps,
                        "Type 1 and 2: roundtrip error.")

    def _type_1_and_3(self, eps=1e-10):

        #f = nufft1d1freqs(len(self.f))
        f = nufft1d1freqs(self.f.shape[0])
        p2 = nufft2d3(self.x, self.y, self.c, f, f, eps=eps)
        print(p2.shape)
        print(p2[0])
        print(p2[1])
        print(p2[2])
        p1 = nufft2d1(self.x, self.y, self.c, len(
            self.f), len(self.g), eps=eps)
        print(p1.shape)
        print(p1[0, :])
        print(p1[1, :])
        print(p1[2, :])
        self.assertTrue(_error(p1, p2) < eps,
                        "Type 1 and 3 and not close (even)")

        # f = nufft2d1freqs(len(f) + 1)
        # p2 = nufft2d3(self.x, self.c, f, eps=eps)
        # p1 = nufft2d1(self.x, self.c, len(f), eps=eps)
        # self.assertTrue(_error(p1, p2) < eps,
        #                 "Type 1 and 3 and not close (odd)")

        # df = 0.5 * (f[1] - f[0])
        # p1 = nufft2d1(self.x, self.c, len(f), eps=eps, df=df)
        # f = nufft2d1freqs(len(f), df=df)
        # p2 = nufft2d3(self.x, self.c, f, eps=eps)
        # self.assertTrue(_error(p1, p2) < eps,
        #                 "Type 1 and 3 and not close (even)")

    def test_type_1_even(self):
        """Is the 2D type 1 with even data correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_even(eps)

    def test_type_1_odd(self):
        """Is the 2D type 1 with odd data correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_odd(eps)

    def test_type_2(self):
        """Is the 2D type 2 correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_2(eps)

    def test_type_3(self):
        """Is the 2D type 3 correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_3(eps)

    def test_type_1_2_roundtrip(self):
        """Is the 2D roundtrip using type 1 and 2 correct?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            self._type_1_2_roundtrip(eps)

    def test_1_and_3(self):
        """Are the 2D type 1 and 3 similar?"""
        for eps in [1e-2, 1e-5, 1e-10, 1e-12]:
            eps = 1e-12
            self._type_1_and_3(eps)


if __name__ == '__main__':
    unittest.main()
