Installation
============

This page is only about how to install ``planktonclas``.

If you want the first practical workflow after installation, use :doc:`quickstart`.

If you want Docker, AI4OS, OSCAR, or the broader project repository, use the companion repository instead:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

Option A: Install from PyPI
---------------------------

Standard package install:

.. code-block:: bash

   pip install planktonclas

For local notebook use:

.. code-block:: bash

   pip install "planktonclas[notebooks]"

What this gives you:

* the ``planktonclas`` command-line tool
* local training and reporting
* local DEEPaaS API usage
* packaged notebook export commands
* the Python modules used by the package

Option B: Development install
-----------------------------

Choose this only if you want to work on the package source itself.

.. code-block:: bash

   git clone https://github.com/lifewatch/planktonclas
   cd planktonclas
   python -m venv .venv
   .venv\Scripts\activate
   pip install -U pip
   pip install -e .

After a repository install, you can also start DEEPaaS directly:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Important notes
---------------

* use ``127.0.0.1`` in the browser; ``0.0.0.0`` is only the bind address
* for local notebooks, install ``"planktonclas[notebooks]"``
* for training and API usage, you will usually create a project first with ``planktonclas init my_project``

Next step
---------

After installation, continue with :doc:`quickstart`.
