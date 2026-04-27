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
* ``pretrained``
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
* published pretrained models are selected through ``pretrained.use_pretrained``, ``pretrained.name``, and ``pretrained.version``
* ``model.modelname`` stays the base architecture choice, while the pretrained selection identifies the published instrument-specific weights to load
* new local training runs save ``best_model.keras`` when validation is enabled; otherwise they save ``final_model.keras``. The published ``FlowCam`` pretrained model currently uses ``final_model.h5`` while ``FlowCyto`` and ``PI10`` are expected to use ``best_model.keras``
* ``planktonclas report`` suggests the most recent timestamp when ``--timestamp`` is omitted and can prompt for another run by number
* ``planktonclas report`` defaults to ``quick`` mode and only generates the subfolder threshold plots in ``full`` mode
* ``planktonclas list-models`` shows published pretrained models with their architecture, version, and checkpoint metadata when the folder name matches a published model id

Practical usage after a model is created
----------------------------------------

Once a model has been trained through the command-line, API, or notebook workflow, you can also interact with it directly from Python.

Typical things you may want to do are:

* load a project config
* load a trained model from a specific timestamp
* predict one image from Python
* call a Dockerized inference server from Python
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
   )

Use a Dockerized inference server from Python
---------------------------------------------

After you have trained a model, reviewed the report, and packaged the run with ``planktonclas docker my_project``, you can talk to the running API from Python with ``requests``.

Start the container, for example:

.. code-block:: bash

   docker run -d -p 5001:5000 --name my-plankton-api my-plankton-api:latest

Then from Python:

.. code-block:: python

   from pathlib import Path

   import requests

   base_url = "http://127.0.0.1:5001"
   health_url = f"{base_url}/api"
   swagger_url = f"{base_url}/swagger.json"
   predict_url = f"{base_url}/v2/models/planktonclas/predict/"

   print(requests.get(health_url, timeout=5).status_code)
   print(requests.get(swagger_url, timeout=5).status_code)

   image_path = Path("example.jpg")
   with image_path.open("rb") as handle:
       response = requests.post(
           predict_url,
           files={"image": (image_path.name, handle, "image/jpeg")},
           timeout=(10, 240),
       )
   response.raise_for_status()
   print(response.json())

This is the same kind of pattern a downstream script can use to:

* ensure the containerized API is available
* inspect ``/swagger.json``
* upload an image for prediction
* parse the returned JSON payload

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
