Reference
=========

Package entry points
--------------------

``planktonclas.api``
   DEEPaaS-facing API layer. Handles metadata, schema generation, training dispatch, model loading, file validation, and prediction formatting.

``planktonclas.train_runfile``
   Direct training runner. Creates output directories, builds generators, trains the TensorFlow model, stores metrics, saves checkpoints, and optionally evaluates a test split.

``planktonclas.config``
   Loads the packaged default config template or a user-provided project ``config.yaml``, validates values, and exposes the flattened configuration dictionary used across the package.

``planktonclas.paths``
   Central path resolver for images, models, checkpoints, logs, stats, and predictions.

``planktonclas.report_utils``
   Generates evaluation plots and summary files in the timestamped ``results/`` directory.

``planktonclas.test_utils``
   Inference helpers for crop-based prediction and top-k accuracy computation.

``planktonclas.visualization``
   Visualization and explainability utilities, including saliency-related helpers used by the notebooks.

Configuration map
-----------------

The runtime configuration is grouped in the active ``config.yaml`` under:

* ``general``
* ``model``
* ``dataset``
* ``training``
* ``monitor``
* ``augmentation``
* ``testing``

Important conventions
---------------------

* images are read from ``general.images_directory``
* if ``data/dataset_files/`` is empty, training can generate split files automatically from the image-folder structure
* if you provide custom split files, ``classes.txt`` and ``train.txt`` are the minimum expected files under ``data/dataset_files/``
* outputs are organized by training timestamp under ``models/<timestamp>/``
* training with test evaluation saves both prediction JSON files and a compact metrics JSON under ``models/<timestamp>/predictions/``
* inference defaults to the latest available trained timestamp
* new local training runs save their final exported model as ``final_model.keras``, while the legacy pretrained ``Phytoplankton_EfficientNetV2B0`` model still uses ``final_model.h5``
* ``planktonclas report`` suggests the most recent timestamp when ``--timestamp`` is omitted and can prompt for another run by number
* ``planktonclas report`` defaults to ``quick`` mode and only generates the subfolder threshold plots in ``full`` mode

Practical usage after a model is created
----------------------------------------

Once a model has been trained through the command-line, API, or notebook workflow, you can also interact with it directly from Python.

Typical things you may want to do are:

* load a project config
* load a trained model from a specific timestamp
* predict one image from Python
* inspect where the package is writing model outputs

Load the project config
-----------------------

.. code-block:: python

   from planktonclas import config

   config.set_config_path("my_project/config.yaml")
   conf = config.get_conf_dict()

Load a trained model
--------------------

.. code-block:: python

   from planktonclas import config, paths
   from planktonclas.api import load_inference_model

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()

   load_inference_model(
       timestamp="2026-03-26_120000",
       ckpt_name="best_model.keras",
   )

Predict one image from Python
-----------------------------

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

Inspect output locations
------------------------

.. code-block:: python

   from planktonclas import config, paths

   config.set_config_path("my_project/config.yaml")
   paths.CONF = config.get_conf_dict()

   print(paths.get_models_dir())
   print(paths.get_checkpoints_dir())
   print(paths.get_logs_dir())
   print(paths.get_predictions_dir())

Source files
------------

For the implementation details, start with these files in the repository:

* ``planktonclas/api.py``
* ``planktonclas/train_runfile.py``
* ``planktonclas/config.py``
* ``planktonclas/paths.py``
* ``planktonclas/test_utils.py``
