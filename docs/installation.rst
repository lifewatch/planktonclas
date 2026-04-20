Installation + Quickstart
=========================

This page is for the package-first workflow.

If you want Docker, AI4OS, OSCAR, or the broader project repository, use the companion repository instead:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

1. Install
----------

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

2. Create a project
-------------------

Create a new project folder:

.. code-block:: bash

   planktonclas init my_project

Or create a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

What this does:

* creates a project-local ``config.yaml``
* creates the expected ``data/`` and ``models/`` folders
* prepares the folder structure used by training, inference, and notebooks

3. Understand the project layout
--------------------------------

After ``planktonclas init``, your project looks like this:

.. code-block:: text

   my_project/
     config.yaml
     data/
       images/
       dataset_files/
     models/

If you also copy the packaged notebooks, your project gains:

.. code-block:: text

   my_project/
     notebooks/

The most important parts are:

* ``config.yaml``: the main user-facing configuration file
* ``data/images/``: your image data, unless you point ``general.images_directory`` somewhere else
* ``data/dataset_files/``: optional split metadata such as ``classes.txt`` and ``train.txt``
* ``models/``: timestamped outputs from training runs

4. Minimal required input
-------------------------

The only mandatory input is the image directory.

You can either:

* put images under ``data/images/``
* or point ``general.images_directory`` in ``config.yaml`` to another folder

If ``data/dataset_files/`` is empty, training can generate split files automatically from the image-folder structure.

If you provide your own metadata files, the expected files are:

* custom-split required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``

5. Common pipeline
------------------

A typical local workflow looks like this:

.. code-block:: bash

   planktonclas init my_project
   planktonclas validate-config --config ./my_project/config.yaml
   planktonclas train --config ./my_project/config.yaml
   planktonclas report --config ./my_project/config.yaml

Optional additions:

.. code-block:: bash

   planktonclas pretrained my_project
   planktonclas api --config ./my_project/config.yaml
   planktonclas notebooks my_project

If you want a quick smoke test with the demo data:

.. code-block:: bash

   planktonclas init my_project --demo
   planktonclas train --config ./my_project/config.yaml --quick
   planktonclas report --config ./my_project/config.yaml

6. What each command means
--------------------------

``planktonclas init my_project``
   Creates a new project folder with the standard structure and a starter ``config.yaml``.

``planktonclas init my_project --demo``
   Creates a runnable demo project so you can test the package quickly.

``planktonclas validate-config --config ./my_project/config.yaml``
   Checks that your configuration is valid before you start training or serving.

``planktonclas train --config ./my_project/config.yaml``
   Runs a training job using the active project config and writes outputs to ``models/<timestamp>/``.

``planktonclas train --config ./my_project/config.yaml --quick``
   Runs a shorter smoke-test training workflow, useful for checking that the setup works.

``planktonclas report --config ./my_project/config.yaml``
   Generates evaluation plots and summary outputs for a training run.

``planktonclas pretrained my_project``
   Downloads the published pretrained model into the project.

``planktonclas api --config ./my_project/config.yaml``
   Starts the local DEEPaaS API so you can use the browser UI or call the prediction/training endpoints.

``planktonclas notebooks my_project``
   Copies the packaged notebooks into your local project folder.

``planktonclas list-models --config ./my_project/config.yaml``
   Lists the available trained model timestamps known to the project.

7. First ways to use it
-----------------------

Local training:

.. code-block:: bash

   planktonclas train --config ./my_project/config.yaml

Local API:

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Packaged notebooks:

.. code-block:: bash

   pip install "planktonclas[notebooks]"
   planktonclas notebooks my_project

8. Notes on outputs
-------------------

Each training run creates a timestamped folder under ``models/``:

.. code-block:: text

   models/<timestamp>/
     ckpts/
     conf/
     logs/
     stats/
     dataset_files/
     predictions/
     results/

Important conventions:

* new local training runs save their final exported model as ``final_model.keras``
* the legacy pretrained ``Phytoplankton_EfficientNetV2B0`` model still uses ``final_model.h5``
* inference usually defaults to the latest available trained timestamp
* ``planktonclas report`` suggests the newest run when ``--timestamp`` is omitted

9. Repository install for development
-------------------------------------

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

Important:

* use ``127.0.0.1`` in the browser; ``0.0.0.0`` is only the bind address
* for prediction, the browser UI supports file uploads for ``image`` and ``zip``
* for training, ``images_directory`` must point to a folder visible to the machine running the API
