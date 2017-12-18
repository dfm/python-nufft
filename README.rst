Python-NUFFT
============

.. image:: https://travis-ci.org/marchdf/python-nufft.svg?branch=master
    :target: https://travis-ci.org/marchdf/python-nufft

.. image:: https://coveralls.io/repos/github/marchdf/python-nufft/badge.svg?branch=master
	   :target: https://coveralls.io/github/marchdf/python-nufft?branch=master

Python bindings to a subset of the `NUFFT algorithm
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. 1D, 2D, and 3D
cases are implemented.

Usage
-----

To install, run ``python setup.py install``. Then, to evaluate a type-3 FT,
use ``nufft.nufft3``. Assuming that you have a time series in ``t`` and ``y``
and you want to evaluate it at (angular) frequencies ``f``:

.. code-block:: python

    import nufft
    ft = nufft.nufft3(t, y, f)

You can specify your required precision using ``eps=1e-15``. The default is
``1e-15``.


Authors and License
-------------------

Python bindings by Dan Foreman-Mackey, Thomas Arildsen, and
Marc T. Henry de Frahan but the code that actually does the work is
from the Greengard lab at NYU (see `the website
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_). The Fortran code
is BSD licensed and the Python bindings are MIT licensed.
