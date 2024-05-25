import subprocess as sp
import sys

def test_module():
    sp.check_output([sys.executable, "-m", "mddatasetbuilder", "-h"])