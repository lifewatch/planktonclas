3. Notebooks Usage
==================

Overview
--------

The repository includes notebooks for:

* dataset exploration
* image transformation
* augmentation
* model training
* prediction
* prediction statistics
* saliency and explainability

They are the best choice when you want an interactive workflow.

The normal package install includes the Python dependencies used by these notebooks. For local notebook use, install the notebook extra in the same environment:

.. code-block:: bash

   pip install "planktonclas[notebooks]"

This extra installs the Jupyter runtime packages needed to open and execute the notebooks locally.

Notebook list
-------------

``1.0-Dataset_exploration.ipynb``
   Explore class balance, dataset composition, and general dataset statistics.

``1.1-Image_transformation.ipynb``
   Inspect and adapt preprocessing so a new dataset matches the expected training input format.

``1.2-Image_augmentation.ipynb``
   Experiment with augmentation strategies.

``2.0-Model_training.ipynb``
   Run model training interactively.

``3.0-Computing_predictions.ipynb``
   Predict one image or many images and inspect raw outputs.

``3.1-Prediction_statistics.ipynb``
   Evaluate predictions on a labeled split and inspect metrics and confusion-style summaries.

``3.2-Saliency_maps.ipynb``
   Visualize explainability outputs.

Finding the notebooks
---------------------

Copy the packaged notebooks into your project with:

.. code-block:: bash

   planktonclas notebooks my_project

This creates ``my_project/notebooks/`` and copies the packaged notebooks there.

To refresh an existing project with updated packaged notebooks:

.. code-block:: bash

   planktonclas notebooks my_project --force

The copied notebooks auto-detect the nearest project ``config.yaml``, so they use the paths inside your local project folder rather than the installed package directory.
They also copy ``data/data_transformation/start``, ``reference_style``, and ``end`` for the image-transformation notebook.

For ``1.1-Image_transformation.ipynb``:

* put your new raw images in ``data/data_transformation/start/``
* keep one or more reference images in ``data/data_transformation/reference_style/``
* the transformed outputs are written to ``data/data_transformation/end/``

For the model-based notebooks ``3.0-Computing_predictions.ipynb``, ``3.1-Prediction_statistics.ipynb``, and ``3.2-Saliency_maps.ipynb``, the most important variables are ``TIMESTAMP`` and ``MODEL_NAME`` near the top of the notebook. They are prefilled for the published pretrained model ``Phytoplankton_EfficientNetV2B0`` so the notebooks run immediately, but you should change them to your own training timestamp and checkpoint name when you want to inspect a newly trained model.

If you are already running Jupyter locally, open that directory and work from there.

If you are inside an AI4OS deployment or a container image that ships the helper commands, you may also have:

.. code-block:: bash

   deep-start -j

That command is deployment-specific. It is not part of the local ``planktonclas`` CLI.

Recommended order
-----------------

1. dataset exploration
2. transformations and augmentation
3. model training
4. predictions
5. prediction statistics
6. saliency maps
