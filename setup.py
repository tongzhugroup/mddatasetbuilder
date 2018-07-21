from setuptools import setup
setup(name='MDDatasetMaker',
      version='1.0.0',
      description='A script to make molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.',
      keywords="molecular dynamics dataset",
      url='https://github.com/njzjz/MDDatasetMaker',
      author='Jinzhe Zeng',
      author_email='njzjz@qq.com',
      packages=['MDDatasetMaker'],
      install_requires=['numpy','scikit-learn','ReacNetGenerator'])

