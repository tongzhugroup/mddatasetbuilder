import numpy as np
from ase import Atoms

from mddatasetbuilder.detect import DetectDump

def test_bond_pbc():
    atoms = Atoms("O2",
                 positions=[[0., 0., 0.], [19., 19., 19.]],
                 pbc=True,
                 cell=np.diag([20., 20., 20.]),
                 )
    bonds = DetectDump._crd2bond(atoms, False)
    assert bonds == [[1], [0]]
    levels = DetectDump._crd2bond(atoms, True)
    assert levels == [[1], [1]]


def test_bond_nopbc():
    atoms = Atoms("O2",
                 positions=[[0., 0., 0.], [19., 19., 19.]],
                 pbc=False,
                 cell=np.diag([20., 20., 20.]),
                 )
    bonds = DetectDump._crd2bond(atoms, False)
    assert bonds == [[], []]
    levels = DetectDump._crd2bond(atoms, True)
    assert levels == [[], []]
