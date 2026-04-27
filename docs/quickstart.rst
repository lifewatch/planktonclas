Quickstart
==========

This page shows the first practical workflow after installation.

If you have not installed the package yet, start with :doc:`installation`.

Quickstart pipeline
-------------------

The common order is:

1. create a project
2. validate the config
3. optionally download the pretrained model
4. train a model
5. generate a report
6. optionally build an inference Docker image

Step 1: Create a project
------------------------

.. code-block:: bash

   planktonclas init my_project

Or create a runnable demo project:

.. code-block:: bash

   planktonclas init my_project --demo

This creates:

* a project-local ``config.yaml``
* a ``data/`` folder
* a ``models/`` folder

Step 2: Validate the config
---------------------------

.. code-block:: bash

   planktonclas validate-config my_project

Step 3: Optional pretrained model
---------------------------------

If you want to start from the published pretrained model:

.. code-block:: bash

   planktonclas pretrained my_project

Step 4: Train a model
---------------------

.. code-block:: bash

   planktonclas train my_project

For a quick smoke test on a demo project:

.. code-block:: bash

   planktonclas train my_project --quick

Step 5: Generate a report
-------------------------

Before packaging a model run into Docker, it is usually best to inspect the report first and confirm that you are happy with the trained run.

.. code-block:: bash

   planktonclas report my_project

If you leave out ``--timestamp``, ``planktonclas report`` suggests the newest run automatically.

Step 6: Optional inference Docker image
---------------------------------------

If you want a more stable packaged inference runtime after training:

.. code-block:: bash

   planktonclas docker my_project

This packages the latest trained model run into a Docker image that serves the API for inference.

You can also select a specific run:

.. code-block:: bash

   planktonclas docker my_project --timestamp 2026-04-21_120000 --ckpt-name best_model.keras --tag my-plankton-api:latest

Project structure
-----------------

After ``planktonclas init``, your project looks like this:

.. code-block:: text

   my_project/
     config.yaml
     data/
       images/
       dataset_files/
     models/

Minimal required input
----------------------

The only mandatory input is the image directory.

You can either:

* put images under ``data/images/``
* or point ``general.images_directory`` in ``config.yaml`` to another folder

If ``data/dataset_files/`` is empty, training can generate split files automatically from the image-folder structure.

If you provide your own metadata files, the expected files are:

* custom-split required: ``classes.txt``, ``train.txt``
* optional: ``val.txt``, ``test.txt``, ``info.txt``, ``aphia_ids.txt``

Typical additions
-----------------

After the main quickstart pipeline, you can also use:

.. code-block:: bash

   planktonclas api my_project
   planktonclas notebooks my_project
   planktonclas list-models my_project

If you keep the standard project layout created by ``planktonclas init``, these commands automatically use ``my_project/config.yaml``. Use ``--config PATH`` only when your config file lives somewhere else.

Next step
---------

Continue with one of these workflow pages:

* :doc:`python_usage`
* :doc:`api_usage`
* :doc:`notebooks`
