Planktonclass: ZooScan
=======================================================


**Authors:** [Wout Decrop](https://github.com/woutdecrop) *(VLIZ)* and [Jonas Mortelmans](https://github.com/jonasmortelmansvliz) *(VLIZ)*

This branch is focused on the **ZooScan processing workflow** at VLIZ.
It uses the published `planktonclass` package for classification and adds the scripts and configuration needed to process ZooScan data locally.

For general package usage, training workflows, CLI commands, API usage, notebooks, and full documentation, see:

- https://phyto-plankton-classification.readthedocs.io/en/latest/


## ZooScan Setup

This branch is intended to **install `planktonclass` as a package** rather than install the package source from this branch.

Create and activate a fresh virtual environment:

```bash
mkdir zooscan_processing
cd zooscan_processing
python -m venv cyto
.\vpi10\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install the package and the extra dependencies needed for the PI10 scripts:

```bash
pip install planktonclass
```



## Scope Of This Branch

This README only documents the ZooScan-specific setup and entry points.
The broader `planktonclass` package documentation, including installation options, training, prediction workflows, notebooks, and deployment guidance, is maintained in the main documentation site:

- https://planktonclass.readthedocs.io/en/latest/


## Acknowledgements

If you use this project, please consider citing the original paper for the package:

> Decrop, W., Lagaisse, R., Mortelmans, J., Muñiz, C., Heredia, I., Calatrava, A., & Deneudt, K. (2025). *Automated image classification workflow for phytoplankton monitoring*. **Frontiers in Marine Science, 12**. https://doi.org/10.3389/fmars.2025.1699781
