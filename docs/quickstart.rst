Quickstart
==========

This quickstart is package-focused.

If you want Docker, AI4OS, OSCAR, or the broader project workflow, go to the companion repository instead:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

Choose your path
----------------

Most package users should choose one of these:

1. local training with the CLI
2. local API usage
3. packaged notebooks inside a local project

Option A: Use it locally
------------------------

.. code-block:: bash

   planktonclas init my_project
   planktonclas validate-config --config ./my_project/config.yaml

For a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

To download the published pretrained model into the project:

.. code-block:: bash

   planktonclas pretrained my_project

Local training
--------------

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

Outputs are written into a timestamped directory under ``my_project/models/``.

For a quick smoke test on the demo project:

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml --quick

Generate a report
-----------------

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

Local API
---------

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Notebook workflow
-----------------

For local notebook use:

.. code-block:: bash

   pip install "planktonclas[notebooks]"

.. code-block:: bash

   planktonclas notebooks my_project

This copies the packaged notebooks into ``my_project/notebooks/``.

Option B: Use API
-----------------

.. code-block:: bash

   pip install planktonclas

.. code-block:: bash

   planktonclas init my_project

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Useful commands
---------------

.. code-block:: bash

   planktonclas list-models --config ./my_project/config.yaml
   planktonclas pretrained my_project

Where to Go for Full Project Topics
-----------------------------------

Use the full repository for:

* Docker image build and runtime instructions
* AI4OS deployment
* OSCAR deployment
* marketplace-facing metadata and assets
* broader phytoplankton workflow explanation

Repository:

* https://github.com/ai4os-hub/phyto-plankton-classification
