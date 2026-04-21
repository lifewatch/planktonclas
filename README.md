Planktonclas: FlowCam
=======================================================

[![Tests](https://github.com/lifewatch/planktonclas/actions/workflows/tests.yml/badge.svg)](https://github.com/lifewatch/planktonclas/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/planktonclas.svg)](https://pypi.org/project/planktonclas/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/planktonclas.svg)](https://pypi.org/project/planktonclas/)

<table>
  <tr>
    <td valign="top">

**Author:** [Wout Decrop](https://github.com/woutdecrop) (VLIZ) 

**Related publication:**  
[*Automated image classification workflow for phytoplankton monitoring*](https://doi.org/10.3389/fmars.2025.1699781)

**Resources:**
- [Documentation](https://plantonclas.readthedocs.io/en/latest/)
- [PyPI package](https://pypi.org/project/planktonclas/)
- [Package downloads](https://pypi.org/project/planktonclas/)

**Projects:** [iMagine](https://www.imagine-ai.eu/)
  
`planktonclas` is a toolkit for training, evaluating, and serving phytoplankton image classifiers!


It was originally developed for FlowCam data, and has also been retrained or adapted in separate branches for other instruments and datasets:

- [FlowCam / main branch](https://github.com/lifewatch/planktonclas/tree/master)
- [Zooscan branch](https://github.com/lifewatch/planktonclas/tree/zooscan)
- [Cyto branch](https://github.com/lifewatch/planktonclas/tree/cyto)
- [PI10 branch](https://github.com/lifewatch/planktonclas/tree/PI10)

If you want the full repository with Docker, OSCAR, AI4OS, packaged deployment assets, and broader project explanation, see:
- [`phyto-plankton-classification`](https://github.com/ai4os-hub/phyto-plankton-classification)



  </td>
    <td valign="top">

<pre>
                 +.
               +:      :==.
              %      .#.
             #:*==* *=
           -+**+*####.
          +********%%.
         +*******#**#+
      ********#%%####+
        .*====+==::=#%%*
        -%**   --::=-:.
        +=#.   -:::+.
-+*++:  +.     +:::*
:+.  .+- ==:   +::::*
=-    == ::-+*+:::::*##-
.+.  :+-.-====-:::::+%#.
 ===*: :++::::-=:++*#=
  -#. -+**:::=*++**%##+
 .=+-=   ##*:**#*%******=
 .=**+  =*++#************#-
         .++*****++++++++*##+
          :+*+#%++++++++*+.
              ***  :###-
            ::#**.  +**+
           .%@+.: --@@@%
                   :.
</pre>

  </td>
  </tr>
</table>

## Install

Install with Python 3.12 and `pip`:

```bash
pip install planktonclas
```

For notebook support:

```bash
pip install "planktonclas[notebooks]"
```


## Choose Your Path

### 1. I want to train locally

Use:

```bash
planktonclas train --config ./my_project/config.yaml
```

This is the best choice if you already know where your image folder is and want a direct local workflow.

### 2. I want to use a browser UI / API

Use:

```bash
planktonclas api --config ./my_project/config.yaml
```

Then open:

- `http://127.0.0.1:5000/ui`
- `http://127.0.0.1:5000/api#/`

This is the best choice if you want to interact through the DEEPaaS UI or integrate with an external service.

### 3. I want notebooks

Use:

```bash
planktonclas notebooks my_project
```

This copies the packaged notebooks into `my_project/notebooks/`. It is the best choice for exploration, augmentation experiments, prediction analysis, and explainability.

`pip install planktonclas` installs the package dependencies used by the notebooks, including TensorFlow, plotting, and reporting libraries.
For local notebook use, install the notebook extra instead:

```bash
pip install "planktonclas[notebooks]"
```

## Quick Start

### Option A: Use it locally
[Read the Docs site](https://plantonclas.readthedocs.io/en/latest/)

```bash
pip install planktonclas
```

Then create a project:

```bash
planktonclas init my_project
```

Or create a runnable demo project:

```bash
planktonclas init my_project --demo
```

*OPTIONAL*: Validate the generated config:

```bash
planktonclas validate-config --config ./my_project/config.yaml
```


Local training:

```bash
planktonclas train --config ./my_project/config.yaml
```

For a quick smoke test on the demo project:

```bash
planktonclas train --config ./my_project/config.yaml --quick
```

*OPTIONAL*: Download the published pretrained model into the project:

```bash
planktonclas pretrained my_project
```

For the bundled legacy pretrained model `Phytoplankton_EfficientNetV2B0`, use the
checkpoint `final_model.h5`. New training runs created by `planktonclas train`
save their final exported model as `final_model.keras`.

Report generation after training:

```bash
planktonclas report --config ./my_project/config.yaml
```

If you leave out `--timestamp`, `planktonclas report` suggests the most recent run, lists the available timestamps, and lets you choose another one by number.
It also lets you choose between `quick` and `full` mode. `quick` is the default and creates the core figures only; `full` also generates the threshold-based plots in the `results/` subfolders.

### Option B: Use api
[Read the Docs site](https://plantonclas.readthedocs.io/en/latest/)

```bash
pip install planktonclas
```

Then create a project:

```bash
planktonclas init my_project
```


Local API:

```bash
planktonclas api --config ./my_project/config.yaml
```

### Option C: I want notebooks

For local notebook use:

```bash
pip install "planktonclas[notebooks]"
```

Then create a project:

```bash
planktonclas init my_project
```

Copy notebooks into the project:

```bash
planktonclas notebooks my_project
```

In the model-based notebooks (`3.0`, `3.1`, and `3.2`), the first variables to check are `TIMESTAMP` and `MODEL_NAME`. They are prefilled for the published pretrained model so the notebooks work out of the box, but when you want to inspect a model from your own training run you should change those two values first.


## Project Structure

After `planktonclas init`, your project looks like this:

```text
my_project/
  config.yaml
  data/
    images/
    dataset_files/
  models/
  notebooks/
```

### What is required?

The only mandatory input is the image directory:

- `data/images/`
- or the directory pointed to by `images_directory` in `config.yaml`

If `data/dataset_files/` is empty, training can generate dataset splits automatically from the image-folder structure.

If you provide your own dataset metadata files, the expected files are:

- custom-split required: `classes.txt`, `train.txt`
- optional: `val.txt`, `test.txt`, `info.txt`, `aphia_ids.txt`

The split files map image paths to integer labels starting at `0`.

## Configuration

The main user config is a project-local `config.yaml`.

It is created by:

```bash
planktonclas init my_project
```

Most users only need to adjust a small number of fields:

- `general.base_directory`
- `general.images_directory`
- `model.modelname`
- `training.epochs`
- `training.batch_size`
- `training.use_validation`
- `training.use_test`
- `monitor.use_tensorboard`

Internal-only values such as model-specific preprocessing are now derived automatically and are not meant to be edited by users.

## Local CLI Workflow

The package installs a `planktonclas` command with these main subcommands:

- `planktonclas init [DIR]`
- `planktonclas init [DIR] --demo`
- `planktonclas validate-config --config PATH`
- `planktonclas train --config PATH`
- `planktonclas report --config PATH [--timestamp TS]`
- `planktonclas api --config PATH`
- `planktonclas pretrained [DIR]`
- `planktonclas list-models --config PATH`
- `planktonclas notebooks [DIR]`

Typical local workflow:

```bash
planktonclas init my_project
planktonclas notebooks my_project
planktonclas validate-config --config ./my_project/config.yaml
planktonclas train --config ./my_project/config.yaml
planktonclas report --config ./my_project/config.yaml
```

For a faster package smoke test with the demo data:

```bash
planktonclas init my_project --demo
planktonclas train --config ./my_project/config.yaml --quick
planktonclas report --config ./my_project/config.yaml
```

## API Workflow

Start the API with:

```bash
planktonclas init my_project
planktonclas api --config ./my_project/config.yaml
```

Then open:

- `http://127.0.0.1:5000/ui`
- `http://127.0.0.1:5000/api#/`

You can also start DEEPaaS directly after a repo install:

```powershell
$env:PLANKTONCLAS_CONFIG = (Resolve-Path .\my_project\config.yaml)
$env:DEEPAAS_V2_MODEL = "planktonclas"
deepaas-run --listen-ip 0.0.0.0
```

Important notes:

- `0.0.0.0` is a bind address, not the browser URL
- open `127.0.0.1` in the browser
- for prediction, the browser UI supports file uploads for `image` and `zip`
- for training, `images_directory` is a path field, so it must point to a folder visible to the machine running the API

## Notebook Workflow


Copy the packaged notebooks into your project with:

```bash
planktonclas init my_project
planktonclas notebooks my_project
```

The copied notebooks auto-detect the nearest project `config.yaml`, so they use the paths inside your local project folder rather than the installed package directory.
They also copy `data/data_transformation/start`, `reference_style`, and `end` for the transformation notebook.

Notebook overview:

- `1.0-Dataset_exploration.ipynb`
- `1.1-Image_transformation.ipynb`
- `1.2-Image_augmentation.ipynb`
- `2.0-Model_training.ipynb`
- `3.0-Computing_predictions.ipynb`
- `3.1-Prediction_statistics.ipynb`
- `3.2-Saliency_maps.ipynb`

For `1.1-Image_transformation.ipynb`:

- put your new raw images in `data/data_transformation/start/`
- keep one or more reference images in `data/data_transformation/reference_style/`
- the transformed outputs are written to `data/data_transformation/end/`

## Outputs

Each training run creates a timestamped folder under `models/`:

```text
models/<timestamp>/
  ckpts/
  conf/
  logs/
  stats/
  dataset_files/
  predictions/
  results/
```

Useful outputs include:

- checkpoints like `best_model.keras`
- `stats.json`
- saved prediction JSON files
- saved test metrics JSON files with top-k accuracy, precision, recall, and F1 summaries
- report images and CSV summaries under `results/`

To generate performance plots after training:

```bash
planktonclas report --config ./my_project/config.yaml
```

## More Documentation

The full documentation is available here:

- [Read the Docs site](https://plantonclas.readthedocs.io/en/latest/)
- [Documentation entry page](docs/index.rst)

Main documentation pages:

- [Installation](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [API usage](docs/api_usage.rst)
- [Python usage](docs/python_usage.rst)
- [Notebooks](docs/notebooks.rst)
- [Reference](docs/reference.rst)

For Docker, OSCAR, AI4OS, and the broader deployment-oriented repository, see:
- https://github.com/ai4os-hub/phyto-plankton-classification


## Development

Choose this only if you want to work on the package itself.

```bash
git clone https://github.com/lifewatch/planktonclas
cd phyto-plankton-classification
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
pip install -e ".[dev]"
python -m pytest
```

## Acknowledgements

If you use this project, please consider citing:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781
