# karaboXAS

## Introduction

A high-level API for performing data analysis for **X-ray Absorption Spectroscopy (XAS)** experiments at SCS. 

The design aims to let the scientist be able to visualize the result with only one line of code. For example:

```py
from karaboXAS import XasTim

fig, ax = XasTim("/gpfs/exfel/exp/SCS/201831/p900048/raw/r0413").process(50).select('XGM', 0.1).select(['MCP1', 'MCP2', 'MCP3'], 1).plot_spectrum(n_bins=40)
```

In addition, it also provides rich interface for performing data analysis and visualization in great details.


## Installation

On the Maxwell cluster:
```sh
$ git clone https://github.com/European-XFEL/karaboXAS.git
$ cd karaboXAS
$ module load anaconda3
$ pip install --user .
```

On the online cluster:
```sh
$ module load anaconda3
$ pip install --user karaboXAS -f /gpfs/exfel/sw/software/wheelhouse --no-index
```

## Tutorials

Tutorials can be found in the following notebooks:

- [XAS with TIM](./notebooks/xas_with_tim_tutorial.ipynb)

In case of problems during launching a Jupyter Notebook, you may need to upgrade `ipykernel` to fix them.

On the `Maxwell` cluster:
```sh
$ module load anaconda3
$ pip install --user --upgrade ipykernel
```

On the online cluster:
```sh
$ module load anaconda3
$ pip install --user --upgrade ipykernel -f /gpfs/exfel/sw/software/wheelhouse --no-index
```