"""The purpose of this script is to test the NUFFT library through the
python-nufft interface. This seeks to validate the NUFFT
implementation for the special case where it corresponds to a DFT.
The script transforms 2-dimensional arrays between the space/time and
the Fourier domain and compares the output to that of the
corresponding FFT implementation from NumPy.

"""

from __future__ import division
import numpy as np
from .nufft import nufft2d1, nufft2d2, nufft2d3

### Parameters ###
num_samples = 32 * np.ones(2, dtype=int)
eps = 1e-13 # This is arbitrarily set not cause any errors further
            # investigation would be needed to determine a reasonable
            # expected error margin

### Time points ###
# Coordinates
x = 2 * np.pi * np.arange(num_samples[0]) / num_samples[0]
y = 2 * np.pi * np.arange(num_samples[1]) / num_samples[1]
xy_grid = np.meshgrid(x,y)
# Values
#c = np.random.uniform(size=num_samples)
c = xy_grid[0] + xy_grid[1]
### Frequency points ###
st_grid = np.meshgrid(np.arange(num_samples[0]), np.arange(num_samples[1]))

### NUFFT type 1 DFT transforms ###
f_dir1 = np.roll(np.roll(nufft2d1(xy_grid[0].reshape(-1),
                                  xy_grid[1].reshape(-1),
                                  c.reshape(-1), num_samples[0],
                                  num_samples[1], iflag=-1,
                                  direct=True),
                         -int(num_samples[0]/2), 0),
                 -int(num_samples[1]/2), 1) * num_samples.prod()
f_nufft1 = np.roll(np.roll(nufft2d1(xy_grid[0].reshape(-1),
                                    xy_grid[1].reshape(-1),
                                    c.reshape(-1),
                                    num_samples[0],
                                    num_samples[1], iflag=-1,
                                    direct=False),
                           -int(num_samples[0]/2), 0),
                   -int(num_samples[1]/2), 1) * num_samples.prod()
f_numpy = np.fft.fft2(c)

assert np.sqrt(np.sum(np.abs(f_dir1 - f_numpy).reshape(-1) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT direct DFT (1) vs. NumPy FFT: error too large"
assert np.sqrt(np.sum(np.abs(f_nufft1 - f_numpy).reshape(-1) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT FFT (1) vs. NumPy FFT: error too large"

### NUFFT type 1 IDFT transforms ###
c_dir = np.roll(np.roll(nufft2d1(xy_grid[0].reshape(-1),
                                 xy_grid[1].reshape(-1),
                                 f_numpy.reshape(-1),
                                 num_samples[0], num_samples[1],
                                 iflag=1, direct=True),
                        -int(num_samples[0]/2), 0),
                -int(num_samples[1]/2), 1)
c_nufft = np.roll(np.roll(nufft2d1(xy_grid[0].reshape(-1),
                                   xy_grid[1].reshape(-1),
                                   f_numpy.reshape(-1),
                                   num_samples[0],
                                   num_samples[1], iflag=1,
                                   direct=False),
                          -int(num_samples[0]/2), 0),
                  -int(num_samples[1]/2), 1)
c_numpy = np.fft.ifft2(f_numpy)

assert np.sqrt(np.sum(np.abs(c_dir - c_numpy).reshape(-1) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT direct IDFT (1) vs. NumPy IFFT: error too large"
assert np.sqrt(np.sum(np.abs(c_nufft - c_numpy).reshape(-1) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT IFFT (1) vs. NumPy IFFT: error too large"

### NUFFT type 2 DFT transforms ###
f_dir2 = nufft2d2(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                  np.roll(np.roll( c, -int(num_samples[0]/2),
                                   0), -int(num_samples[1]/2), 1), iflag=-1,
                  direct=True)
f_nufft2 = nufft2d2(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                    np.roll(np.roll( c, -int(num_samples[0]/2),
                                     0), -int(num_samples[1]/2), 1), iflag=-1,
                    direct=False)

assert np.sqrt(np.sum(np.abs(f_dir2 - f_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT direct DFT (2) vs. NumPy FFT: error too large"
assert np.sqrt(np.sum(np.abs(f_nufft2 - f_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT FFT (2) vs. NumPy FFT: error too large"

### NUFFT type 2 IDFT transforms ###
c_dir2 = nufft2d2(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                  np.roll(np.roll( f_numpy,
                                   -int(num_samples[0]/2), 0),
                          -int(num_samples[1]/2), 1), iflag=1,
                  direct=True) / num_samples.prod()
c_nufft2 = nufft2d2(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                    np.roll(np.roll( f_numpy,
                                     -int(num_samples[0]/2), 0),
                            -int(num_samples[1]/2), 1), iflag=1,
                    direct=False) / num_samples.prod()

assert np.sqrt(np.sum(np.abs(c_dir2 - c_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large"
assert np.sqrt(np.sum(np.abs(c_nufft2 - c_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT IFFT (2) vs. NumPy IFFT: error too large"

### NUFFT type 3 DFT transforms ###
f_dir3 = nufft2d3(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                  c.reshape(-1), st_grid[0].reshape(-1),
                  st_grid[1].reshape(-1), iflag=-1,
                  direct=True) * num_samples.prod()
f_nufft3 = nufft2d3(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                    c.reshape(-1), st_grid[0].reshape(-1),
                    st_grid[1].reshape(-1), iflag=-1,
                    direct=False) * num_samples.prod()

assert np.sqrt(np.sum(np.abs(f_dir3 - f_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT direct DFT (3) vs. NumPy FFT: error too large"
assert np.sqrt(np.sum(np.abs(f_nufft3 - f_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(f_numpy).reshape(-1)**2)) < eps, "NUFFT FFT (3) vs. NumPy FFT: error too large"

#### NUFFT type 3 IDFT transforms ###
c_dir3 = nufft2d3(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                  f_numpy.reshape(-1), st_grid[0].reshape(-1),
                  st_grid[1].reshape(-1), iflag=1, direct=True)
c_nufft3 = nufft2d3(xy_grid[0].reshape(-1), xy_grid[1].reshape(-1),
                    f_numpy.reshape(-1), st_grid[0].reshape(-1),
                    st_grid[1].reshape(-1), iflag=1,
                    direct=False)

assert np.sqrt(np.sum(np.abs(c_dir3 - c_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT direct IDFT (2) vs. NumPy IFFT: error too large"
assert np.sqrt(np.sum(np.abs(c_nufft3 - c_numpy.reshape(-1)) ** 2) /
               np.sum(np.abs(c_numpy).reshape(-1)**2)) < eps, "NUFFT IFFT (2) error vs. NumPy IFFT: error too large"
