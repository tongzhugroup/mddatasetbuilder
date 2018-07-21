# MDDatasetMaker

A script to make molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.

Author: Jinzhe Zeng

Email: njzjz@qq.com  10154601140@stu.ecnu.edu.cn

[Research Group](http://computchem.cn)

## Simple example

A [LAMMPS bond file](http://lammps.sandia.gov/doc/fix_reax_bonds.html) and a [LAMMPS dump file](https://lammps.sandia.gov/doc/dump.html) should be prepared.

```python
>>> from MDDatasetMaker import DatasetMaker
>>> DatasetMaker(bondfilename='bonds.reaxc.ch4_new',dumpfilename='dump.ch4',dataset_dir='dataset_ch4',xyzfilename='ch4',stepinterval=25).makedataset()
```
