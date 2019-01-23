# MDDatasetBuilder
[![python3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://badge.fury.io/py/MDDatasetBuilder)

MDDatasetBuilder is a script to build molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.

**Author**: Jinzhe Zeng

**Email**: jzzeng@stu.ecnu.edu.cn

[Research Group](http://computchem.cn)

## Requirements
* Python 3.6
* Python packages: [numpy](https://github.com/numpy/numpy), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [ASE](https://gitlab.com/ase/ase), [GaussianRunner](https://github.com/njzjz/gaussianrunner), [tqdm](https://github.com/tqdm/tqdm)
* Software: [Gaussian 16](http://gaussian.com/), [DeePMD](https://github.com/deepmodeling/deepmd-kit)

## Installation

```sh
$ git clone https://github.com/njzjz/mddatasetbuilder
$ cd mddatasetbuilder/
$ python3 setup.py install
```

You can test whether ReacNetGenerator is running normally:
```sh
% python3 setup.py test
```

## Simple example

A [LAMMPS bond file](http://lammps.sandia.gov/doc/fix_reax_bonds.html) and a [LAMMPS dump file](https://lammps.sandia.gov/doc/dump.html) should be prepared.

```bash
$ datasetbuilder -d dump.ch4 -b bonds.reaxc.ch4_new -a C H O -n ch4 -i 25
```

Then you can calculate generated Gaussian files:

```bash
$ qmcalc -d dataset_ch4_GJf/000
$ qmcalc -d dataset_ch4_GJf/001
```

Next, prepare DeePMD datas and use [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) to train a model.

```bash
$ preparedeepmd -p dataset_ch4_GJf -a C H O
$ cd train && dp_train train.json
```