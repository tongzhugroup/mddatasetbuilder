# MDDatasetBuilder

[![python version](https://img.shields.io/pypi/pyversions/mddatasetbuilder.svg?logo=python&logoColor=white)](https://pypi.org/project/mddatasetbuilder)
[![PyPI](https://img.shields.io/pypi/v/mddatasetbuilder.svg)](https://pypi.org/project/mddatasetbuilder)
[![Build Status](https://travis-ci.com/njzjz/mddatasetbuilder.svg?branch=master)](https://travis-ci.com/njzjz/mddatasetbuilder)
[![Coverage Status](https://coveralls.io/repos/github/njzjz/mddatasetbuilder/badge.svg?branch=master)](https://coveralls.io/github/njzjz/mddatasetbuilder?branch=master)
[![codecov](https://codecov.io/gh/njzjz/mddatasetbuilder/branch/master/graph/badge.svg)](https://codecov.io/gh/njzjz/mddatasetbuilder)
[![Research Group](https://img.shields.io/website-up-down-green-red/http/computchem.cn.svg?label=Research%20Group)](http://computchem.cn)

MDDatasetBuilder is a script to build molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.

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
