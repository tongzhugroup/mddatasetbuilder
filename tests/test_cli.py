"""Test command line interface."""

import subprocess as sp
import sys


def test_module():
    """Test python -m mddatasetbuilder."""
    sp.check_output([sys.executable, "-m", "mddatasetbuilder", "-h"])
