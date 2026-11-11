Installation
############

CCpy is currently run and tested on Linux and Mac OS devices. Linux users (including WSL users)
can choose to install a pre-compiled version of CCpy from the PyPI server (simplest option) or
download the source code and install it manually. For now, Mac OS users must download and install
the source code (wheels for Mac OS will be uploaded to PyPI in the near future).

Installing from PyPI
--------------------
For Linux machines, the
latest version of CCpy available on PyPI is obtained by running ::

    pip install coupled-cluster-py

Installing via Source Code
--------------------------

Clone the CCpy repository and enter the :code:`ccpy` directory: ::

    git clone https://piecuch-group/ccpy.git
    cd ccpy

We recommend creating a new environment for CCpy by running the following command ::

    conda create --name=ccpy_env python=3.12

and installing all of the dependencies listed in :code:`requirements-dev.txt` via ::

    pip install -r requirements-dev.txt

Additionally, it is useful to install :code:`cmake` and :code:`pkgconfig` specific to your
Conda environment by running ::

    conda install pkgconfig cmake

Then, you can install CCpy using ::

    pip install --no-build-isolation --verbose --editable .

The Meson backend will automatically locate the needed libraries with the help of :code:`pkgconfig`.
If you are having issues finding :code:`openblas`, make sure that the environment variable :code:`PKG_CONFIG_PATH` points to
the directory that includes the :code:`openblas.pc` file. This should be located within :code:`openblas/lib`,
or something similar. After installing in editable mode (via :code:`--editable`), the package will
automatically update with any changes you make without additional installation steps.
