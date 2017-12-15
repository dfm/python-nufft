#!/usr/bin/env python

import os
import sys
from numpy.distutils.core import setup, Extension

# Hackishly inject a constant into builtins to enable importing of the
# package even if numpy isn't installed. Only do this if we're not
# running the tests!
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__NUFFT_SETUP__ = True
import nufft
version = nufft.__version__

# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()

# Set up the compiled extension.
sources = list(map(os.path.join("src", "nufft1d", "{0}").format,
                   ["dfftpack.f", "dirft1d.f", "dirft2d.f", "dirft3d.f",
                    "next235.f", "nufft1df90.f", "nufft2df90.f", "nufft3df90.f"]))
sources += [os.path.join("nufft", "nufft1d.pyf")]
extensions = [Extension("nufft._nufft", sources=sources)]

setup(
    name="nufft",
    version=version,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    url="https://github.com/dfm/python-nufft",
    license="MIT",
    packages=["nufft"],
    install_requires=[
        'numpy',
        'sphinx_rtd_theme'
    ],
    ext_modules=extensions,
    description="non-uniform FFTs",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE"]},
    test_suite='tests',
    tests_require=['nose'],
    include_package_data=True,
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
