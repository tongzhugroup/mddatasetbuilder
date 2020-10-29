# MDDatasetBuilder

[![DOI:10.1039/C9CP05091D](https://zenodo.org/badge/DOI/10.1038/s41467-020-19497-z.svg)](https://doi.org/10.1038/s41467-020-19497-z)
[![python version](https://img.shields.io/pypi/pyversions/mddatasetbuilder.svg?logo=python&logoColor=white)](https://pypi.org/project/mddatasetbuilder)
[![PyPI](https://img.shields.io/pypi/v/mddatasetbuilder.svg)](https://pypi.org/project/mddatasetbuilder)
[![codecov](https://codecov.io/gh/njzjz/mddatasetbuilder/branch/master/graph/badge.svg)](https://codecov.io/gh/njzjz/mddatasetbuilder)
[![Research Group](https://img.shields.io/website-up-down-green-red/https/computchem.cn.svg?label=Research%20Group)](https://computchem.cn)

MDDatasetBuilder is a script to construct reference datasets for the training of neural network potentials from given LAMMPS trajectories.

Complex Reaction Processes in Combustion Unraveled by Neural Network Based Molecular Dynamics Simulation, Nature Communications (in press), DOI: [10.1038/s41467-020-19497-z](https://doi.org/10.1038/s41467-020-19497-z)

**Author**: Jinzhe Zeng

**Email**: jzzeng@stu.ecnu.edu.cn

## Installation

Firstly, the latest version of [Anaconda or Miniconda](https://conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html) shoule be installed. And then use conda to install [openbabel](https://github.com/openbabel/openbabel):

```sh
conda install openbabel -c conda-forge
```

Then install mddatasetbuilder can be installed with pip:
```
pip install git+https://github.com/tongzhugroup/mddatasetbuilder
```

The installation process should be very quick, taking only a few minutes on a “normal” desktop computer. 

## Usage
### Simple example

A [LAMMPS dump file](https://lammps.sandia.gov/doc/dump.html) should be prepared. A [LAMMPS bond file](http://lammps.sandia.gov/doc/fix_reax_bonds.html) can be added for the addition information.

```bash
datasetbuilder -d dump.ch4 -b bonds.reaxc.ch4_new -a C H O -n ch4 -i 25
```

Here, `dump.ch4` is the name of the dump file. `bonds.reaxc.ch4_new` is the name of the bond file, which is optional. `C H O` is the element in the trajectory. `ch4` is the name of the dataset. `25` means the time step interval and the default value is 1.

Then you can generate Gaussian input files for each structure in the dataset and calculate the potential energy & atomic forces (assume the Gaussian 16 has already been installed.):

```bash
qmcalc -d dataset_ch4_GJf/000
qmcalc -d dataset_ch4_GJf/001
```

Next, prepare a DeePMD dataset and use [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) to train a NN model.

```bash
preparedeepmd -p dataset_ch4_GJf -a C H O
cd train && dp train train.json
```

The runtime of the software depends on the amount of data. It is more suited to running on a server rather than desktop computer.

### DP-GEN
The MDDatasetBuilder package has been integrated with [DP-GEN](https://github.com/deepmodeling/dpgen) software.
