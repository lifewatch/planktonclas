1. Python Usage
===============

Overview
--------

You can use the package directly from Python without going through the DEEPaaS web UI. The most relevant modules are:

* ``planktonclas.config``: load and validate configuration
* ``planktonclas.paths``: resolve data, model, log, and output directories
* ``planktonclas.train_runfile``: run training
* ``planktonclas.api``: load trained models and run prediction logic
* ``planktonclas.test_utils``: prediction helpers used by inference

Load a project config
---------------------

.. code-block:: python

   from planktonclas import config

   config.set_config_path("my_project/config.yaml")
   conf = config.get_conf_dict()

   print(conf["general"]["images_directory"])
   print(conf["training"]["epochs"])

The default template is shipped inside the package, but normal user workflows should point to a project-local ``config.yaml``.

Run training from Python
------------------------

.. code-block:: python

   from datetime import datetime
   from planktonclas import config, paths
   from planktonclas.train_runfile import train_fn

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()

   conf = config.get_conf_dict()
   conf["training"]["epochs"] = 5

   timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
   train_fn(TIMESTAMP=timestamp, CONF=conf)

This creates a new timestamped output directory under ``my_project/models/``.

Inspect output paths
--------------------

.. code-block:: python

   from planktonclas import config, paths

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()

   print(paths.get_models_dir())
   print(paths.get_checkpoints_dir())
   print(paths.get_logs_dir())

Be aware that ``paths.timestamp`` controls which timestamped run directory is currently addressed.

Load a trained model for inference
----------------------------------

.. code-block:: python

   from planktonclas import config, paths
   from planktonclas.api import load_inference_model

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()
   load_inference_model(timestamp="2026-03-26_120000", ckpt_name="best_model.keras")

Run prediction from Python
--------------------------

The direct inference helper used by the API is ``planktonclas.test_utils.predict``. A typical flow is:

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

Practical caution
-----------------

For scripted use, keep the same generated project layout as ``planktonclas init`` unless you are deliberately overriding paths in the configuration.
