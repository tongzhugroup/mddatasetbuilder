# MDDatasetBuilder

[![python version](https://img.shields.io/pypi/pyversions/mddatasetbuilder.svg?logo=python&logoColor=white)](https://pypi.org/project/mddatasetbuilder)
[![PyPI](https://img.shields.io/pypi/v/mddatasetbuilder.svg)](https://pypi.org/project/mddatasetbuilder)
[![Build Status](https://travis-ci.com/njzjz/mddatasetbuilder.svg?branch=master)](https://travis-ci.com/njzjz/mddatasetbuilder)
[![Build status](https://ci.appveyor.com/api/projects/status/70v6eefoe8mgdjtu?svg=true)](https://ci.appveyor.com/project/jzzeng/mddatasetbuilder)
[![Coverage Status](https://coveralls.io/repos/github/njzjz/mddatasetbuilder/badge.svg?branch=master)](https://coveralls.io/github/njzjz/mddatasetbuilder?branch=master)
[![codecov](https://codecov.io/gh/njzjz/mddatasetbuilder/branch/master/graph/badge.svg)](https://codecov.io/gh/njzjz/mddatasetbuilder)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/924ce85b916040079fb453785ec673f9)](https://www.codacy.com/app/jzzeng/mddatasetbuilder?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=njzjz/mddatasetbuilder&amp;utm_campaign=Badge_Grade)

MDDatasetBuilder is a script to build molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.

**Author**: Jinzhe Zeng

**Email**: jzzeng@stu.ecnu.edu.cn

[![Research Group](https://img.shields.io/website-up-down-green-red/http/computchem.cn.svg?label=Research%20Group)](http://computechem.cn)

## Requirements
* Python 3.6 - 3.7
* Python packages:
[numpy](https://github.com/numpy/numpy),
[scikit-learn](https://github.com/scikit-learn/scikit-learn),
[ASE](https://gitlab.com/ase/ase),
[GaussianRunner](https://github.com/njzjz/gaussianrunner),
[tqdm](https://github.com/tqdm/tqdm),
[coloredlogs](https://github.com/xolox/python-coloredlogs),
[lz4](https://github.com/python-lz4/python-lz4),
[pybase64](https://github.com/mayeut/pybase64)
* Software:
[Gaussian 16](http://gaussian.com/),
[DeePMD](https://github.com/deepmodeling/deepmd-kit)

## Installation

```sh
git clone https://github.com/njzjz/mddatasetbuilder
cd mddatasetbuilder
pip install .
```

You can test whether ReacNetGenerator is running normally:
```sh
python3 setup.py pytest
```

## Simple example

A [LAMMPS bond file](http://lammps.sandia.gov/doc/fix_reax_bonds.html) and a [LAMMPS dump file](https://lammps.sandia.gov/doc/dump.html) should be prepared.

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
cd train && dp_train train.json
```