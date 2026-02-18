=====
bfast
=====

The bfast package provides a highly-efficient parallel implementation for the `Breaks For Additive Season and Trend (BFASTmonitor) <https://bfast.r-forge.r-project.org>`_ proposed by Verbesselt et al. The implementation is based on `OpenCL <https://www.khronos.org/opencl>`_.

================
Fork Information
================

This repository is a fork of `diku-dk/bfast:master <https://github.com/diku-dk/bfast>`_. It includes several updates to support modern Python environments and dependencies:

**Build System Migration**

- Migrated to `setuptools` and fixed deprecated API usage.
- Replaced deprecated `numpy.distutils` with standard `setuptools` in `setup.py`.
- Updated `requirements.txt` to support modern library versions (NumPy >= 1.24, Pandas >= 2.0).

**Compatibility Fixes**

Addressed compatibility issues arising from dependency upgrades:

- Replaced removed NumPy types (`np.int`, `np.float`, `np.bool`) with native Python types.
- Replaced deprecated `np.warnings` with the standard `warnings` module.
- Updated Sphinx configuration to use `add_css_file`.
- Fixed dictionary iteration (`iteritems` -> `items`) and Matplotlib usage.

=============
Documentation
=============

See the `documentation <https://bfast.readthedocs.io>`_ for details and examples.

============
Dependencies
============

The bfast package has been tested under Python 3.8+. The required Python dependencies are:

- numpy>=1.24
- pandas>=2.0
- pyopencl>=2022.1
- scikit-learn>=1.0
- scipy>=1.10
- matplotlib>=3.7
- wget>=3.2

For building the documentation, the following additional dependencies are required:

- Sphinx>=7.0
- sphinx-bootstrap-theme>=0.8
- numpydoc>=1.5

Further, `OpenCL <https://www.khronos.org/opencl>`_ needs to be available for the OpenCL backend.

==========
Quickstart
==========

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/petescarth/bfast.git
  cd bfast

Afterwards, on Linux systems, you can install the package locally for the current user via::

  pip install .

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv3). The authors are not responsible for any implications that stem from the use of this software.

