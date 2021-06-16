"""Detect from trajectory."""
import pickle
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from collections import defaultdict

try:
    from openbabel import openbabel
except ImportError:
    import openbabel
import numpy as np
from ase import Atom, Atoms
from scipy.spatial import cKDTree
from .dps import dps as connectmolecule


class Detect(metaclass=ABCMeta):
    def __init__(self, filename, atomname, pbc, errorlimit=None, errorfilename=None):
        self.filename = filename
        self.atomname = atomname
        self.pbc = pbc
        self.errorlimit = errorlimit
        self.errorfilename = errorfilename
        self.steplinenum = self._readN()

    @abstractmethod
    def _readN(self):
        pass

    @abstractmethod
    def readatombondtype(self, item):
        """This function reads bond types of atoms such as C1111."""
        pass

    @abstractmethod
    def readmolecule(self, lines):
        """This function reads molecules."""
        pass

    @staticmethod
    def gettype(inputtype):
        """Get the class for the input file type."""
        if inputtype == 'bond':
            detectclass = DetectBond
        elif inputtype == 'dump':
            detectclass = DetectDump
        else:
            raise RuntimeError("Wrong input file type")
        return detectclass


class DetectBond(Detect):
    """LAMMPS bond file."""

    def _readN(self):
        """Read bondfile N, which should be at very beginning."""
        # copy from reacnetgenerator on 2018-12-15
        with open(self.filename if isinstance(self.filename, str) else self.filename[0]) as f:
            iscompleted = False
            for index, line in enumerate(f):
                if line.startswith("#"):
                    if line.startswith("# Number of particles"):
                        if iscompleted:
                            stepbindex = index
                            break
                        else:
                            iscompleted = True
                            stepaindex = index
                        N = [int(s) for s in line.split() if s.isdigit()][0]
                        atomtype = np.zeros(N, dtype=np.int)
                else:
                    s = line.split()
                    atomtype[int(s[0])-1] = int(s[1])
        steplinenum = stepbindex-stepaindex
        self._N = N
        self.atomtype = atomtype
        self.atomnames = self.atomname[self.atomtype-1]
        return steplinenum

    def readatombondtype(self, item):
        # copy from reacnetgenerator on 2018-12-15
        (step, lines), _ = item
        d = defaultdict(list)
        for line in lines:
            if line:
                if line[0] != "#":
                    s = line.split()
                    atombond = sorted(
                        map(lambda x: max(1, round(float(x))), s[4 + int(s[2]): 4 + 2 * int(s[2])]))
                    d[pickle.dumps((self.atomnames[int(s[0]) - 1],
                                    atombond))].append(int(s[0]))
        return d, step

    def readmolecule(self, lines):
        """Returns molecules from lines.

        Parameters
        ----------
        lines: list of strs
            Lines of LAMMPS bond files.

        Returns
        -------
        molecules: list
            Indexes of atoms in molecules.
        """
        # copy from reacnetgenerator on 2018-12-15
        bond = [None]*self._N
        for line in lines:
            if line:
                if not line.startswith("#"):
                    s = line.split()
                    bond[int(s[0])-1] = map(lambda x: int(x) -
                                            1, s[3:3+int(s[2])])
        molecules = connectmolecule(bond)
        return molecules


class DetectDump(Detect):
    def _readN(self):
        # copy from reacnetgenerator on 2018-12-15
        iscompleted = False
        with open(self.filename if isinstance(self.filename, str) else self.filename[0]) as f:
            for index, line in enumerate(f):
                if line.startswith("ITEM:"):
                    linecontent = self.LineType.linecontent(line)
                    if linecontent == self.LineType.ATOMS:
                        keys = line.split()
                        self.id_idx = keys.index('id') - 2
                        self.tidx = keys.index('type') - 2
                        self.xidx = keys.index('x') - 2
                        self.yidx = keys.index('y') - 2
                        self.zidx = keys.index('z') - 2
                else:
                    if linecontent == self.LineType.NUMBER:
                        if iscompleted:
                            stepbindex = index
                            break
                        else:
                            iscompleted = True
                            stepaindex = index
                        N = int(line.split()[0])
                        atomtype = np.zeros(N, dtype=int)
                    elif linecontent == self.LineType.ATOMS:
                        s = line.split()
                        atomtype[int(s[0])-1] = int(s[1])
        steplinenum = stepbindex-stepaindex
        self._N = N
        self.atomtype = atomtype
        self.atomnames = self.atomname[self.atomtype-1]
        return steplinenum

    def readatombondtype(self, item):
        (step, lines), needlerror = item
        if needlerror:
            trajline, errorline = lines
            lerror = np.fromstring(errorline, dtype=float, sep=' ')[7:]
        d = defaultdict(list)
        step_atoms, ids = self.readcrd(lines)
        if needlerror:
            lerror = [x for (y, x) in sorted(zip(ids, lerror))]
        level = self._crd2bond(step_atoms, readlevel=True)
        for i, (n, l) in enumerate(zip(self.atomnames, level)):
            if not needlerror or lerror[i] > self.errorlimit:
                # Note that atom id starts from 1
                d[pickle.dumps((n, sorted(l)))].append(i+1)
        return d, step

    def readmolecule(self, lines):
        """Returns molecules from lines.

        Parameters
        ----------
        lines: list of strs
            Lines of LAMMPS bond files.

        Returns
        -------
        molecules: list
            Indexes of atoms in molecules.
        step_atoms: ase.Atoms
            The atoms of the frame.
        """
        step_atoms, _ = self.readcrd(lines)
        bond = self._crd2bond(step_atoms, readlevel=False)
        molecules = connectmolecule(bond)
        # return atoms as well
        return molecules, step_atoms

    @classmethod
    def _crd2bond(cls, step_atoms, readlevel):
        # copy from reacnetgenerator on 2019/4/13
        # updated on 2019/10/11
        atomnumber = len(step_atoms)
        if step_atoms.pbc.any():
            # Apply period boundry conditions
            # add ghost atoms
            repeated_atoms = step_atoms.repeat(2)[atomnumber:]
            tree = cKDTree(step_atoms.get_positions())
            d = tree.query(repeated_atoms.get_positions(), k=1)[0]
            nearest = d < 5
            ghost_atoms = repeated_atoms[nearest]
            realnumber = np.where(nearest)[0] % atomnumber
            step_atoms += ghost_atoms
        # Use openbabel to connect atoms
        mol = openbabel.OBMol()
        mol.BeginModify()
        for idx, (num, position) in enumerate(zip(step_atoms.get_atomic_numbers(), step_atoms.positions)):
            a = mol.NewAtom(idx)
            a.SetAtomicNum(int(num))
            a.SetVector(*position)
        mol.ConnectTheDots()
        if not readlevel:
            bond = [[] for i in range(atomnumber)]
        else:
            mol.PerceiveBondOrders()
            bondlevel = [[] for i in range(atomnumber)]
        mol.EndModify()
        for b in openbabel.OBMolBondIter(mol):
            s1 = b.GetBeginAtom().GetId()
            s2 = b.GetEndAtom().GetId()
            if s1 >= atomnumber and s2 >= atomnumber:
                # duplicated
                continue
            elif s1 >= atomnumber:
                s1 = realnumber[s1-atomnumber]
            elif s2 >= atomnumber:
                s2 = realnumber[s2-atomnumber]
            if not readlevel:
                bond[s1].append(s2)
                bond[s2].append(s1)
            else:
                level = b.GetBondOrder()
                bondlevel[s1].append(level)
                bondlevel[s2].append(level)
        return bondlevel if readlevel else bond

    def readcrd(self, item):
        """Only this function can read coordinates."""
        lines = item
        boxsize = []
        step_atoms = []
        ids = []
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    linecontent = self.LineType.linecontent(line)
                else:
                    if linecontent == self.LineType.ATOMS:
                        s = line.split()
                        ids.append(int(s[self.id_idx]))
                        step_atoms.append(Atom(
                            self.atomname[int(s[self.tidx]) - 1],
                            (float(s[self.xidx]), float(s[self.yidx]), float(s[self.zidx]))))
                    elif linecontent == self.LineType.BOX:
                        s = line.split()
                        boxsize.append(float(s[1])-float(s[0]))
        # sort by ID
        step_atoms = [x for (y, x) in sorted(zip(ids, step_atoms))]
        step_atoms = Atoms(step_atoms, cell=boxsize, pbc=self.pbc)
        return step_atoms, ids

    class LineType(Enum):
        """Line type in the LAMMPS dump files."""

        TIMESTEP = auto()
        ATOMS = auto()
        NUMBER = auto()
        BOX = auto()
        OTHER = auto()

        @classmethod
        def linecontent(cls, line):
            """Return line content."""
            if line.startswith("ITEM: TIMESTEP"):
                return cls.TIMESTEP
            if line.startswith("ITEM: ATOMS"):
                return cls.ATOMS
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                return cls.NUMBER
            if line.startswith("ITEM: BOX"):
                return cls.BOX
            return cls.OTHER
