"""Use 'pip install .' to install mddatasetbuilder."""

import os

from setuptools import Extension, setup
from wheel.bdist_wheel import bdist_wheel


class bdist_wheel_abi3(bdist_wheel):
    """Build the distributed wheel for abi3."""

    def get_tag(self):
        """Get the wheel tag."""
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.7
            return "cp37", "abi3", plat

        return python, abi, plat


if __name__ == "__main__":
    define_macros = [("CYTHON_LIMITED_API", "1"), ("Py_LIMITED_API", "0x03070000")]
    if os.environ.get("DEBUG", 0):
        define_macros.extend((("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")))
    setup(
        ext_modules=[
            Extension(
                "mddatasetbuilder.dps",
                sources=["mddatasetbuilder/dps.pyx", "mddatasetbuilder/c_stack.cpp"],
                language="c++",
                define_macros=define_macros,
                py_limited_api=True,
            ),
        ],
        cmdclass={"bdist_wheel": bdist_wheel_abi3},
    )
