from setuptools import setup
setup(name='mddatasetbuilder',
      description='A script to make molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.',
      keywords="molecular dynamics dataset",
      url='https://github.com/njzjz/mddatasetbuilder',
      author='Jinzhe Zeng',
      author_email='jzzeng@stu.ecnu.edu.cn',
      packages=['mddatasetbuilder'],
      install_requires=['numpy', 'scikit-learn', 'ase',
                        'gaussianrunner', 'tqdm', 'coloredlogs'
                        ],
      entry_points={
          'console_scripts': ['datasetbuilder=mddatasetbuilder.datasetbuilder:_commandline',
                              'qmcalc=mddatasetbuilder.qmcalc:_commandline',
                              'preparedeepmd=mddatasetbuilder.deepmd:_commandline'
                              ]
      },
      test_suite='mddatasetbuilder.test',
      tests_require=['requests'],
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      )
