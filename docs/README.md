# MDDatasetBuilder

[![DOI:10.1038/s41467-020-19497-z](https://img.shields.io/badge/DOI-10.1038%2Fs41467--020--19497--z-blue)](https://doi.org/10.1038/s41467-020-19497-z)
[![python version](https://img.shields.io/pypi/pyversions/mddatasetbuilder.svg?logo=python&logoColor=white)](https://pypi.org/project/mddatasetbuilder)
[![PyPI](https://img.shields.io/pypi/v/mddatasetbuilder.svg)](https://pypi.org/project/mddatasetbuilder)
[![codecov](https://codecov.io/gh/njzjz/mddatasetbuilder/branch/master/graph/badge.svg)](https://codecov.io/gh/njzjz/mddatasetbuilder)
[![Research Group](https://img.shields.io/website-up-down-green-red/https/computchem.cn.svg?label=Research%20Group)](https://computchem.cn)

MDDatasetBuilder is a script to construct reference datasets for the training of neural network potentials from given LAMMPS trajectories.

Complex reaction processes in combustion unraveled by neural network-based molecular dynamics simulation, Nature Communications, **11**, 5713 (2020), DOI: [10.1038/s41467-020-19497-z](https://doi.org/10.1038/s41467-020-19497-z)

**Author**: Jinzhe Zeng

**Email**: jzzeng@stu.ecnu.edu.cn

## Installation

MDDatasetBuilder can be installed with pip:

```
pip install mddatasetbuilder
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
The MDDatasetBuilder package has been integrated with [DP-GEN](https://github.com/deepmodeling/dpgen) software:
```bash
dpgen init_reaction reaction.json machine.json
```
where an example of `reaction.json` can be found [here](https://github.com/deepmodeling/dpgen/blob/v0.9.1/examples/init/reaction.json), and `machine.json` should include the following keys:
`reaxff_command`, `reaxff_resources`, `reaxff_machine`, `build_command`, `build_resources`, `build_machine`, `fp_command`, `fp_resources`, `fp_machine`, and `fp_group_size`.
`reaxff_command` is the LAMMPS command, `build_command` is the MDDatasetbuilder command, and `fp_command` is the Gaussian 16 command.
