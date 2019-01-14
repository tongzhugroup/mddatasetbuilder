from setuptools import setup
setup(name='mddatasetbuilder',
      version='1.0.12',
      description='A script to make molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.',
      keywords="molecular dynamics dataset",
      url='https://github.com/njzjz/mddatasetbuilder',
      author='Jinzhe Zeng',
      author_email='jzzeng@stu.ecnu.edu.cn',
      packages=['mddatasetbuilder'],
      install_requires=['numpy', 'scikit-learn', 'ase', 'gaussianrunner'],
      entry_points={
          'console_scripts': ['datasetbuilder=mddatasetbuilder.datasetbuilder:_commandline',
                              'qmcalc=mddatasetbuilder.qmcalc:_commandline',
                              'preparedeepmd=mddatasetbuilder.deepmd:_commandline'
                              ]
      })
