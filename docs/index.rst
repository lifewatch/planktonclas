Planktonclas
============

``planktonclas`` is the package repository for training, evaluating, and serving phytoplankton image classifiers.

It is the package-first home for:

* local installation
* project initialization
* training and reporting
* local Python usage
* local DEEPaaS API usage
* packaged notebook workflows

What the package does
---------------------

With ``planktonclas``, you can:

* create a standard project structure with ``planktonclas init``
* train a classifier from a project ``config.yaml``
* generate reports from a training run
* start a local DEEPaaS API for browser-based training and prediction
* copy packaged notebooks into a local project
* work with a published pretrained model

Typical workflow
----------------

For most users, the common order is:

1. install the package
2. create a project
3. validate the config
4. train a model
5. generate a report
6. optionally continue with Python, API, or notebook-based usage

Which page to start with
------------------------

Start with:

* :doc:`installation` if you only want to install the package
* :doc:`quickstart` if you want the shortest working pipeline

Then continue with one of these workflows:

* :doc:`python_usage` for direct Python-based usage
* :doc:`api_usage` for DEEPaaS API usage
* :doc:`notebooks` for notebook-based usage
* :doc:`reference` for the package reference and internal conventions

Companion repository
--------------------

If you want the full repository with Docker, OSCAR, AI4OS, packaged deployment assets, and broader project explanation, use the companion repository:

* ``phyto-plankton-classification``: https://github.com/ai4os-hub/phyto-plankton-classification

Citation
--------

If you use this package, please consider citing:

* Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Package Guide

   installation
   quickstart
   python_usage
   api_usage
   notebooks
   reference
