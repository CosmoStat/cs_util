Installation
============

.. note::

  If your package name on PyPi differs from the default package name you
  will need to update the URL in ``installation.rst`` accordingly.

Users
-----

You can install the latest release from `PyPi <https://pypi.org/project/cs_util/>`_
as follows:

.. code-block:: bash

  pip install cs_util


Alternatively clone the repository and build the package locally as follows:

.. code-block:: bash

  pip install .


Developers
----------

Developers are recommend to clone the repository and build the package locally
in development mode with testing and documentation packages as follows:

.. code-block:: bash

  pip install -e ".[develop]"
