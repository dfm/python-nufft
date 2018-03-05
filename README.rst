Python-NUFFT
============

.. image:: https://travis-ci.org/ThomasA/python-nufft.svg?branch=master
    :target: https://travis-ci.org/ThomasA/python-nufft

.. image:: https://coveralls.io/repos/github/ThomasA/python-nufft/badge.svg?branch=master
           :target: https://coveralls.io/github/ThomasA/python-nufft?branch=master

.. image:: https://readthedocs.org/projects/python-nufft/badge/?version=latest
	   :target: http://python-nufft.readthedocs.io/en/latest/?badge=latest
	   :alt: Documentation Status

Python bindings to a subset of the `NUFFT algorithm
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. 1D, 2D, and 3D
cases are implemented.

Usage
-----

The documentation can be found on `ReadTheDocs
<https://python-nufft.readthedocs.io/en/latest/>`_.

To install, run ``python setup.py install``. Then, to evaluate a
type-3 FT in 1D, use ``nufft.nufft1d3``. Assuming that you have a time
series in ``t`` and ``y`` and you want to evaluate it at (angular)
frequencies ``f``:

.. code-block:: python

    import nufft
    ft = nufft.nufft1d3(t, y, f)

You can specify your required precision using ``eps=1e-15``. The
default is ``1e-15``.

Authors and License
-------------------

Python bindings by Dan Foreman-Mackey, Thomas Arildsen, and
Marc T. Henry de Frahan but the code that actually does the work is
from the Greengard lab at NYU (see `the website
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_). The Fortran code
is BSD licensed and the Python bindings are MIT licensed.
