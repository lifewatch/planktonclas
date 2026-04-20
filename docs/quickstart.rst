Quickstart
==========

This page is the shortest path to a working local setup.

If you want Docker, AI4OS, OSCAR, or the broader project workflow, go to the companion repository instead:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

If you want the more detailed setup notes and command explanations, see :doc:`installation`.

Fastest local pipeline
----------------------

Install the package:

.. code-block:: bash

   pip install planktonclas

Create a project:

.. code-block:: bash

   planktonclas init my_project

Validate the config:

.. code-block:: bash

   planktonclas validate-config --config ./my_project/config.yaml

Train:

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

Generate a report:

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

That is the basic local workflow.

Quick demo pipeline
-------------------

If you want to test the package quickly without preparing your own dataset first:

.. code-block:: bash

   planktonclas init my_project --demo
   planktonclas train --config ./my_project/config.yaml --quick
   planktonclas report --config ./my_project/config.yaml

Use the local API
-----------------

If you want a browser UI instead of starting with training:

.. code-block:: bash

   pip install planktonclas
   planktonclas init my_project
   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Use notebooks
-------------

If you want the packaged notebooks:

.. code-block:: bash

   pip install "planktonclas[notebooks]"
   planktonclas init my_project
   planktonclas notebooks my_project

This copies the notebooks into ``my_project/notebooks/``.

Typical command order
---------------------

For most users, the common order is:

.. code-block:: bash

   pip install planktonclas
   planktonclas init my_project
   planktonclas validate-config --config ./my_project/config.yaml
   planktonclas train --config ./my_project/config.yaml
   planktonclas report --config ./my_project/config.yaml

If needed, you can add:

.. code-block:: bash

   planktonclas pretrained my_project
   planktonclas api --config ./my_project/config.yaml
   planktonclas notebooks my_project

Where to read more
------------------

Use :doc:`installation` for:

* what each command means
* project structure
* required input files
* output folders
* development install

Use the full repository for:

* Docker image build and runtime instructions
* AI4OS deployment
* OSCAR deployment
* marketplace-facing metadata and assets
* broader phytoplankton workflow explanation

Repository:

* https://github.com/ai4os-hub/phyto-plankton-classification
