"""Use 'pip install .' to install mddatasetbuilder."""


import os

from setuptools import setup, Extension

if __name__ == "__main__":
    define_macros = []
    if os.environ.get("DEBUG", 0):
        define_macros.extend((("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")))
    setup(
        ext_modules=[
            Extension(
                "mddatasetbuilder.dps",
                sources=["mddatasetbuilder/dps.pyx", "mddatasetbuilder/c_stack.cpp"],
                language="c++",
                define_macros=define_macros,
            ),
        ],
    )
