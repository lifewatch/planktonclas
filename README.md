Planktonclas: FlowCam
=======================================================

`planktonclas` is a toolkit for training, evaluating, and serving phytoplankton image classifiers.

It supports:
- local training from a project config
- browser and service-based use through a DEEPaaS API
- interactive notebook workflows
- containerized execution with Docker
- hosted deployment through AI4OS and OSCAR

**Author:** [Wout Decrop](https://github.com/woutdecrop) (VLIZ) 

**Projects:**
- [iMagine](https://www.imagine-ai.eu/)
- [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/)
- [AI4OS](https://docs.ai4os.eu/en/latest/)

**Marketplace and deployment links:**
- [AI4OS / iMagine Marketplace](https://dashboard.cloud.imagine-ai.eu/marketplace/)
- [AI4OS training and deployment docs](https://docs.ai4os.eu/en/latest/)
- [OSCAR manual deployment guide](https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar-manual.html)
- [OSCAR scripted deployment guide](https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar.html)

**Related publication:**  
[*Automated image classification workflow for phytoplankton monitoring*](https://doi.org/10.3389/fmars.2025.1699781)

## What This Repository Gives You

Use this repository or package when you want to do one of these:

1. Train a phytoplankton classifier on your own image folders.
2. Run predictions on single images or batches of images.
3. Start a local API for browser-based testing or integration.
4. Explore the workflow in Jupyter notebooks.
5. Deploy the model through AI4OS or OSCAR.

The important thing for new users is this:

- you do **not** have to use every workflow
- local training, API usage, and notebooks are **alternative entry points**
- they all use the same package and the same project structure

## Workflow Overview

![Workflow overview](references/Flowchart_github_plankton.drawio.png)

The repository supports five main approaches:

| Approach | Best for | Main command |
| --- | --- | --- |
| Local CLI | straightforward training and reporting | `planktonclas train --config ...` |
| Local API | browser-based testing through Swagger / DEEPaaS | `planktonclas api --config ...` |
| Notebooks | interactive exploration and debugging | `planktonclas notebooks my_project` |
| Docker | isolated reproducible runtime | `docker run ...` |
| AI4OS / OSCAR | hosted or remote deployment | deployment-specific |

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

### 4. I want a containerized environment

Use Docker. This is useful when you want an environment that is closer to deployment, or when you want notebook and API tooling already available inside the image.

### 5. I want hosted deployment

Use AI4OS or OSCAR. This is useful when you want a remote API, remote notebooks, or a managed deployment flow.

## Quick Start

### Option A: Install as a package

[Read the Docs site](https://phyto-plankton-classification.readthedocs.io/)

```bash
pip install planktonclas
```

For local notebook use:

```bash
pip install "planktonclas[notebooks]"
```

Then create a project:

```bash
planktonclas init my_project
```

Or create a runnable demo project:

```bash
planktonclas init my_project --demo
```

Validate the generated config:

```bash
planktonclas validate-config --config ./my_project/config.yaml
```

At that point you can choose:

Local training:

```bash
planktonclas train --config ./my_project/config.yaml
```

For a quick smoke test on the demo project:

```bash
planktonclas train --config ./my_project/config.yaml --quick
```

Local API:

```bash
planktonclas api --config ./my_project/config.yaml
```

Copy notebooks into the project:

```bash
planktonclas notebooks my_project
```

To refresh a project with updated packaged notebooks:

```bash
planktonclas notebooks my_project --force
```

This also copies the transformation workspace used by `1.1-Image_transformation.ipynb` into `my_project/data/data_transformation/`.

Download the published pretrained model into the project:

```bash
planktonclas pretrained my_project
```

For the bundled legacy pretrained model `Phytoplankton_EfficientNetV2B0`, use the
checkpoint `final_model.h5`. New training runs created by `planktonclas train`
save their final exported model as `final_model.keras`.

In the model-based notebooks (`3.0`, `3.1`, and `3.2`), the first variables to check are `TIMESTAMP` and `MODEL_NAME`. They are prefilled for the published pretrained model so the notebooks work out of the box, but when you want to inspect a model from your own training run you should change those two values first.

Report generation after training:

```bash
planktonclas report --config ./my_project/config.yaml
```

If you leave out `--timestamp`, `planktonclas report` suggests the most recent run, lists the available timestamps, and lets you choose another one by number.
It also lets you choose between `quick` and `full` mode. `quick` is the default and creates the core figures only; `full` also generates the threshold-based plots in the `results/` subfolders.

### Option B: Use Docker

This is the simplest repository-based workflow if you want the project files but do not want to install all Python dependencies on your machine.

```bash
git clone https://github.com/ai4os-hub/phyto-plankton-classification
cd phyto-plankton-classification
docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" ai4oshub/phyto-plankton-classification:latest /bin/bash
```

Inside the container, you can use the same `planktonclas` CLI workflow.

### Option C: Repository install for development

Choose this only if you want to work on the package itself.

```bash
git clone https://github.com/ai4os-hub/phyto-plankton-classification
cd phyto-plankton-classification
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

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
planktonclas pretrained my_project
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

The repository includes notebooks for:

- dataset exploration
- preprocessing and image transformation
- augmentation
- model training
- prediction
- prediction statistics
- saliency and explainability

Copy the packaged notebooks into your project with:

```bash
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

## Docker Workflow

Run the published Docker image:

```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" ai4oshub/phyto-plankton-classification:latest /bin/bash
```

Inside the container, you can use the same `planktonclas` CLI workflow.

The container image also downloads the published pretrained model during the image build, so it is ready inside `models/` without an extra manual download step.

If the image or deployment provides AI4OS helper scripts, you may also have:

```bash
deep-start -j
deep-start --deepaas
```

Important:

- a normal local install does **not** provide `deep-start`
- for local installs, use `planktonclas ...` commands or `deepaas-run`
- `deep-start` is typically available only in AI4OS/container/deployment environments that ship those helpers

## AI4OS and OSCAR

For hosted execution and deployment, see:

- [AI4OS / iMagine Marketplace](https://dashboard.cloud.imagine-ai.eu/marketplace/)
- [AI4OS docs](https://docs.ai4os.eu/en/latest/)
- [OSCAR manual deployment guide](https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar-manual.html)
- [OSCAR scripted deployment guide](https://docs.ai4eosc.eu/en/latest/howtos/deploy/oscar.html)
- [Marketplace-specific notes](references/README_marketplace.md)

The repository also contains deployment assets under `oscar/`, including:

- `oscar/phyto-plankton-classifier.yaml`

## More Documentation

The full documentation is available here:

- [Read the Docs site](https://phyto-plankton-classification.readthedocs.io/)
- [Documentation entry page](docs/index.rst)

Main documentation pages:

- [Installation](docs/installation.rst)
- [Quickstart](docs/quickstart.rst)
- [API usage](docs/api_usage.rst)
- [Python usage](docs/python_usage.rst)
- [Notebooks](docs/notebooks.rst)
- [Reference](docs/reference.rst)

## Acknowledgements

If you use this project, please consider citing:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781

and:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692.
