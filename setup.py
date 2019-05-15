"""Use 'pip install .' to install mddatasetbuilder."""


from os import path

from setuptools import find_packages, setup, Extension

if __name__ == '__main__':
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'docs', 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    tests_require = ['requests', 'pytest-sugar', 'pytest-cov'],
    setup(name='mddatasetbuilder',
          description='A script to make molecular dynamics (MD) datasets for neural networks from given LAMMPS trajectories automatically.',
          keywords="molecular dynamics dataset",
          url='https://github.com/njzjz/mddatasetbuilder',
          author='Jinzhe Zeng',
          author_email='jzzeng@stu.ecnu.edu.cn',
          packages=find_packages(),
          python_requires='~=3.6',
          install_requires=['numpy', 'scikit-learn', 'ase',
                            'gaussianrunner>=1.0.20',
                            'tqdm', 'coloredlogs',
                            'pybase64', 'lz4',
                            ],
          entry_points={
              'console_scripts': ['datasetbuilder=mddatasetbuilder.datasetbuilder:_commandline',
                                  'qmcalc=mddatasetbuilder.qmcalc:_commandline',
                                  'preparedeepmd=mddatasetbuilder.deepmd:_commandline'
                                  ]
          },
          test_suite='mddatasetbuilder.test',
          tests_require=tests_require,
          extras_require={
              "test": tests_require,
          },
          use_scm_version=True,
          setup_requires=[
              'setuptools>=18.0',
              'setuptools_scm',
              'pytest-runner',
              'cython',
          ],
          package_data={
              'mddatasetbuilder': ['test/test.json'],
          },
          long_description=long_description,
          long_description_content_type='text/markdown',
          classifiers=[
              "Natural Language :: English",
              "Operating System :: POSIX :: Linux",
              "Operating System :: Microsoft :: Windows",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Topic :: Scientific/Engineering :: Chemistry",
              "Topic :: Software Development :: Libraries :: Python Modules",
              "Topic :: Software Development :: Version Control :: Git",
          ],
          zip_safe=True,
          ext_modules=[
              Extension("mddatasetbuilder.dps", sources=[
                        "mddatasetbuilder/dps.pyx", "mddatasetbuilder/c_stack.cpp"], language="c++"),
          ],
          )
