Python bindings to a subset of the `NUFFT algorithm
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. Only the 1D/type 3 case is
implemented but it's way faster than Lomb-Scargle!

Usage
-----

To install, run ``python setup.py install``. Then, there is only one function:
``nufft``. Assuming that you have a time series in ``t`` and ``y`` and you
want to evaluate it at (angular) frequencies ``f``:

.. code-block:: python

    import nufft
    nufft.nufft(t, y, f)

You can specify your required precision using ``eps=1e-10``. The default is
``1e-10``.


Authors and License
-------------------

Python bindings by Dan Foreman-Mackey but the code that actually does the work
is from the Greengard lab at NYU (see `the website
<http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_). For reasons of
compatibility, the whole project is GPLv2 licensed.
