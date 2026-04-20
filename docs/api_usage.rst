2. API Usage
============

Overview
--------

The DEEPaaS entry point is defined in ``pyproject.toml``:

.. code-block:: text

   [project.entry-points."deepaas.v2.model"]
   planktonclas = "planktonclas.api"

The simplest way to start the API is:

.. code-block:: bash

   planktonclas api --config ./my_project/config.yaml

Then open:

* ``http://127.0.0.1:5000/ui``
* ``http://127.0.0.1:5000/api#/``

Use ``127.0.0.1`` in the browser. ``0.0.0.0`` is only the bind address.

Direct DEEPaaS startup
----------------------

After a repository install, you can also start the API directly:

.. code-block:: powershell

   $env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
   $env:DEEPAAS_V2_MODEL = "planktonclas"
   deepaas-run --listen-ip 0.0.0.0

Main API functions
------------------

The main public API functions are:

* ``get_metadata()``
* ``get_train_args()``
* ``train(**args)``
* ``get_predict_args()``
* ``predict(**args)``

Training through the API
------------------------

Typical browser flow:

1. start ``planktonclas api --config ./my_project/config.yaml``
2. open ``/ui`` or ``/api#/``
3. find the ``TRAIN`` operation
4. edit the parameters you want
5. execute the request

The most important training parameters are:

* ``images_directory``
* ``modelname``
* ``image_size``
* ``batch_size``
* ``epochs``
* ``use_validation``
* ``use_test``
* ``use_best_model``

Important limitation:

* ``images_directory`` is a path field, not a browser folder picker
* the API cannot open a server-side folder chooser through Swagger UI
* for local use, it is usually better to set the path in ``config.yaml`` before starting the API

Prediction through the API
--------------------------

The prediction endpoint accepts:

* ``image``: a single uploaded image
* ``zip``: a ZIP archive containing one or more images

Typical browser flow:

1. start the API
2. open ``/ui`` or ``/api#/``
3. find the ``PREDICT`` ``POST`` method
4. click ``Try it out``
5. provide either ``image`` or ``zip``
6. click ``Execute``

Prediction response
-------------------

The response contains:

* ``filenames``
* ``pred_lab``
* ``pred_prob``
* ``aphia_ids`` when available

Runtime behavior
----------------

* prediction writes a JSON artifact to the configured predictions directory
* ZIP prediction extracts the archive to a temporary directory and scans recursively for images
* training validates ``images_directory`` before starting
* if there are no models yet, the API can still be used for training, but not for inference
