1. CMD Usage
============

Overview
--------

This page shows the command-line workflow as a sequence of practical steps.

Use this path when you want to work mainly with the ``planktonclas`` commands from the terminal.

CMD workflow
------------

The common order is:

1. install the package
2. create a project
3. validate the project config
4. optionally download the pretrained model
5. train a model
6. generate a report
7. continue with prediction, API usage, notebooks, or model inspection

Step 1: Install the package
---------------------------

.. code-block:: bash

   pip install planktonclas

Step 2: Create a project
------------------------

.. code-block:: bash

   planktonclas init my_project

This creates:

* a project-local ``config.yaml``
* a ``data/`` folder
* a ``models/`` folder

For a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

Step 3: Validate the config
---------------------------

.. code-block:: bash

   planktonclas validate-config --config ./my_project/config.yaml

This is the easiest way to catch path or configuration problems before training.

Step 4: Optional pretrained model
---------------------------------

If you want to start from the published pretrained model:

.. code-block:: bash

   planktonclas pretrained my_project

Step 5: Train a model
---------------------

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

For a quick smoke test on a demo project:

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml --quick

This creates a new timestamped output directory under ``my_project/models/``.

Step 6: Generate a report
-------------------------

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

What the report step creates
----------------------------

The report step writes figures and summary files under ``models/<timestamp>/results/``.

Typical outputs include:

* overview performance figures
* class-based evaluation plots
* CSV summary files
* additional threshold-based plots when you use full mode

Important note:

* ``quick`` mode creates the core report figures
* ``full`` mode also creates the threshold-based plots in the ``results/`` subfolders

If you leave out ``--timestamp``, ``planktonclas report`` suggests the newest run automatically.

Step 7: What you can do after training
--------------------------------------

Once a model has been created, you can continue in several directions.

You can:

* inspect the checkpoints under ``models/<timestamp>/ckpts/``
* inspect logs under ``models/<timestamp>/logs/``
* inspect stats under ``models/<timestamp>/stats/``
* inspect saved predictions under ``models/<timestamp>/predictions/``
* inspect reports under ``models/<timestamp>/results/``
* start the API with ``planktonclas api --config ./my_project/config.yaml``
* copy notebooks with ``planktonclas notebooks my_project``
* list available trained runs with ``planktonclas list-models --config ./my_project/config.yaml``

Useful command summary
----------------------

.. code-block:: bash

   planktonclas init my_project
   planktonclas init my_project --demo
   planktonclas validate-config --config ./my_project/config.yaml
   planktonclas pretrained my_project
   planktonclas train --config ./my_project/config.yaml
   planktonclas train --config ./my_project/config.yaml --quick
   planktonclas report --config ./my_project/config.yaml
   planktonclas list-models --config ./my_project/config.yaml

Practical caution
-----------------

For most users, it is best to keep the standard project layout created by ``planktonclas init`` unless you deliberately want to override paths in ``config.yaml``.
