Testing
=======

Testing is done with the `unittest` framework. In the parent directory, build the source files

.. code-block:: bash

   $ python setup.py build_src build_ext --inplace

and run the testing utility

.. code-block:: bash

   $ nosetests
