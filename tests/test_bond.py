"""Test detecting bonds."""

import numpy as np
from ase import Atoms

from mddatasetbuilder.detect import DetectDump


def test_bond_pbc():
    """Test detecting bonds under PBC condition."""
    atoms = Atoms(
        "O2",
        positions=[[0.0, 0.0, 0.0], [19.0, 19.0, 19.0]],
        pbc=True,
        cell=np.diag([20.0, 20.0, 20.0]),
    )
    bonds = DetectDump._crd2bond(atoms, False)
    assert bonds == [[1], [0]]
    levels = DetectDump._crd2bond(atoms, True)
    assert levels == [[1], [1]]


def test_bond_nopbc():
    """Test detecting bonds under non-PBC condition."""
    atoms = Atoms(
        "O2",
        positions=[[0.0, 0.0, 0.0], [19.0, 19.0, 19.0]],
        pbc=False,
        cell=np.diag([20.0, 20.0, 20.0]),
    )
    bonds = DetectDump._crd2bond(atoms, False)
    assert bonds == [[], []]
    levels = DetectDump._crd2bond(atoms, True)
    assert levels == [[], []]
