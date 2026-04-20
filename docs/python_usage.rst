1. Python Usage
===============

Overview
--------

This page shows the Python-based workflow as a sequence of practical steps.

Use this path when you want to work in Python code instead of mainly using the CLI or the DEEPaaS browser UI.

Python workflow
---------------

The common order is:

1. install the package
2. create a project
3. validate the project config
4. optionally download the pretrained model
5. train a model
6. inspect the outputs written to ``models/``
7. optionally use Python code for custom inference or automation

Step 1: Install the package
---------------------------

.. code-block:: bash

   pip install planktonclas

What this gives you:

* the package itself
* the ``planktonclas`` CLI
* the Python modules used for training and inference

Step 2: Create a project
------------------------

.. code-block:: bash

   planktonclas init my_project

This creates:

* a project-local ``config.yaml``
* a ``data/`` folder
* a ``models/`` folder

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

Step 6: Inspect the outputs
---------------------------

After training, the important outputs are typically:

* checkpoints under ``models/<timestamp>/ckpts/``
* logs under ``models/<timestamp>/logs/``
* stats under ``models/<timestamp>/stats/``
* predictions under ``models/<timestamp>/predictions/``
* reports under ``models/<timestamp>/results/``

You can also generate report figures with:

.. code-block:: bash

   planktonclas report --config ./my_project/config.yaml

Step 7: Use Python directly
---------------------------

Once the project exists and a model is available, you can use Python directly for automation, scripting, or custom inference.

-----------------------------------

.. code-block:: python

   from planktonclas import config, paths, api, test_utils

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()

   api.load_inference_model()
   conf = config.conf_dict

   labels, probabilities = test_utils.predict(
       model=api.model,
       X=["/absolute/path/to/image.png"],
       conf=conf,
       top_K=5,
       filemode="local",
       merge=False,
       use_multiprocessing=False,
   )

Most relevant modules
---------------------

* ``planktonclas.config``: load and validate configuration
* ``planktonclas.paths``: resolve data, model, log, and output directories
* ``planktonclas.train_runfile``: run training
* ``planktonclas.api``: load trained models and run prediction logic
* ``planktonclas.test_utils``: prediction helpers used by inference

Practical caution
-----------------

For scripted use, keep the same generated project layout as ``planktonclas init`` unless you are deliberately overriding paths in the configuration.
