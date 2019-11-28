# MDDatasetBuilder

[![arxiv:1910.12690](http://img.shields.io/badge/arXiv-1911.12252-B31B1B.svg?maxAge=86400)](https://arxiv.org/abs/1911.12252)
[![python version](https://img.shields.io/pypi/pyversions/mddatasetbuilder.svg?logo=python&logoColor=white)](https://pypi.org/project/mddatasetbuilder)
[![PyPI](https://img.shields.io/pypi/v/mddatasetbuilder.svg)](https://pypi.org/project/mddatasetbuilder)
[![codecov](https://codecov.io/gh/njzjz/mddatasetbuilder/branch/master/graph/badge.svg)](https://codecov.io/gh/njzjz/mddatasetbuilder)
[![Research Group](https://img.shields.io/website-up-down-green-red/http/computchem.cn.svg?label=Research%20Group)](http://computchem.cn)

MDDatasetBuilder is a script to build molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.

Neural Network Based in Silico Simulation of Combustion Reactions, arXiv:1911.12252

**Author**: Jinzhe Zeng

**Email**: jzzeng@stu.ecnu.edu.cn

## Installation

```sh
pip install git+https://github.com/njzjz/mddatasetbuilder
```

## Simple example

A [LAMMPS dump file](https://lammps.sandia.gov/doc/dump.html) should be prepared. A [LAMMPS bond file](http://lammps.sandia.gov/doc/fix_reax_bonds.html) can be added for the addition information.

```bash
datasetbuilder -d dump.ch4 -b bonds.reaxc.ch4_new -a C H O -n ch4 -i 25
```

Then you can calculate generated Gaussian files:

```bash
qmcalc -d dataset_ch4_GJf/000
qmcalc -d dataset_ch4_GJf/001
```

Next, prepare DeePMD datas and use [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) to train a model.

```bash
preparedeepmd -p dataset_ch4_GJf -a C H O
cd train && dp train train.json
```
