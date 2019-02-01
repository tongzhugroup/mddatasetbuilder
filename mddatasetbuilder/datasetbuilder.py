"""MDDatasetBuilder.

Run 'datasetbuilder -h' for more details.
"""

__author__ = "Jinzhe Zeng"
__email__ = "jzzeng@stu.ecnu.edu.cn"
__update__ = '2019-02-01'
__date__ = '2018-07-18'
__version__ = '1.1.4'

import argparse
import base64
import gc
import itertools
import logging
import os
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool, Semaphore, cpu_count

import numpy as np
from ase import Atom, Atoms
from ase.data import atomic_numbers
from ase.io import write as write_xyz
from pkg_resources import DistributionNotFound, get_distribution
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


class DatasetBuilder(object):
    """Dataset Builder."""

    def __init__(
            self, atomname=None,
            clusteratom=None, bondfilename="bonds.reaxc",
            dumpfilename="dump.reaxc", dataset_name="md", cutoff=5,
            stepinterval=1, n_clusters=10000,
            qmkeywords="%nproc=4\n#mn15/6-31g(d,p)", nproc=None, pbc=True,
            fragment=True):
        """Init the builder."""
        print(__doc__)
        print(f"Author:{__author__}  Email:{__email__}")
        self.dumpfilename = dumpfilename
        self.bondfilename = bondfilename
        self.dataset_dir = f"dataset_{dataset_name}"
        self.xyzfilename = dataset_name
        self.atomname = atomname if atomname else ["C", "H", "O"]
        self.clusteratom = clusteratom if clusteratom else atomname
        self.atombondtype = []
        self.stepinterval = stepinterval
        self.nproc = nproc if nproc else cpu_count()
        self.cutoff = cutoff
        self.n_clusters = n_clusters
        self.writegjf = True
        self.gjfdir = f'{self.dataset_dir}_gjf'
        self.qmkeywords = qmkeywords
        self.pbc = pbc
        self.fragment = fragment
        self._coulumbdiag = dict(
            ((symbol, atomic_numbers[symbol]**2.4/2) for symbol in atomname))
        self._nstructure = 0

    def builddataset(self, writegjf=True):
        """Build a dataset."""
        self.writegjf = writegjf
        timearray = [time.time()]
        with tempfile.TemporaryDirectory() as self.trajatom_dir:
            for runstep in range(3):
                if runstep == 0:
                    self._readtimestepsbond()
                elif runstep == 1:
                    self.steplinenum = self._readlammpscrdN()
                    with open(os.path.join(self.trajatom_dir, 'chooseatoms'), 'wb') as f:
                        for bondtype in self.atombondtype:
                            self._writecoulumbmatrix(bondtype, f)
                            gc.collect()
                elif runstep == 2:
                    self._mkdir(self.dataset_dir)
                    if self.writegjf:
                        self._mkdir(self.gjfdir)
                    self._writexyzfiles()
                gc.collect()
                timearray.append(time.time())
                logging.info(
                    f"Step {len(timearray)-1} Done! Time consumed (s): {timearray[-1]-timearray[-2]:.3f}")

    @classmethod
    def _produce(cls, semaphore, producelist, parameter):
        for item in producelist:
            semaphore.acquire()
            yield item, parameter

    def _readlammpscrdstep(self, item):
        (_, lines), _ = item
        boxsize = []
        step_atoms = []
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    linecontent = 4 if line.startswith("ITEM: TIMESTEP") else (3 if line.startswith(
                        "ITEM: ATOMS") else (1 if line.startswith("ITEM: NUMBER OF ATOMS") else 2))
                else:
                    if linecontent == 3:
                        s = line.split()
                        step_atoms.append(
                            (int(s[0]),
                             Atom(
                                 self.atomname[int(s[1]) - 1],
                                 [float(x) for x in s[2: 5]])))
                    elif linecontent == 2:
                        s = line.split()
                        boxsize.append(float(s[1])-float(s[0]))
        # sort by ID
        _, step_atoms = zip(*sorted(step_atoms, key=lambda a: a[0]))
        step_atoms = Atoms(step_atoms, cell=boxsize, pbc=self.pbc)
        return step_atoms

    def _writecoulumbmatrix(self, trajatomfilename, fc):
        self.dstep = {}
        with open(os.path.join(self.trajatom_dir, f"stepatom.{trajatomfilename}"), 'rb') as f:
            for line in f:
                s = self._decompress(line).split()
                self.dstep[int(s[0])] = [int(x) for x in s[1].split(",")]
        n_atoms = sum(len(x) for x in self.dstep.values())
        if n_atoms > self.n_clusters:
            # undersampling
            max_counter = Counter()
            stepatom = np.zeros((n_atoms, 2), dtype=np.int32)
            feedvector = np.zeros((n_atoms, 0))
            vector_elements = defaultdict(list)
            with open(self.dumpfilename) as f, Pool(self.nproc, maxtasksperchild=10000) as pool:
                semaphore = Semaphore(360)
                results = pool.imap_unordered(
                    self._writestepmatrix, self._produce(
                        semaphore,
                        enumerate(
                            itertools.islice(
                                itertools.zip_longest(
                                    *[f] * self.steplinenum),
                                0, None, self.stepinterval)),
                        None),
                    100)
                j = 0
                for result in tqdm(
                        results, desc=trajatomfilename, total=self._nstep,
                        unit="timestep"):
                    for stepatoma, vector, symbols_counter in result:
                        stepatom[j] = stepatoma
                        for element in (
                                symbols_counter - max_counter).elements():
                            vector_elements[element].append(
                                feedvector.shape[1])
                            feedvector = np.pad(
                                feedvector, ((0, 0),
                                             (0, 1)),
                                'constant',
                                constant_values=(0, self._coulumbdiag
                                                 [element]))
                        feedvector[j,
                                   sum(
                                       [vector_elements[symbol][: size]
                                        for symbol,
                                        size in symbols_counter.items()],
                                       [])] = vector
                        max_counter |= symbols_counter
                        j += 1
                    semaphore.release()
                logging.info(
                    f"Max counter of {trajatomfilename} is {max_counter}")
            pool.close()
            choosedindexs = self._clusterdatas(
                np.sort(feedvector), n_clusters=self.n_clusters)
            pool.join()
        else:
            stepatom = [(u, vv) for u, v in self.dstep.items() for vv in v]
            choosedindexs = range(n_atoms)
        fc.write(
            self._compress(
                ';'.join(
                    (' '.join((str(x) for x in stepatom[index]))
                     for index in choosedindexs))))
        self._nstructure += len(choosedindexs)

    def _writestepmatrix(self, item):
        (step, _), _ = item
        results = []
        if step in self.dstep:
            step_atoms = self._readlammpscrdstep(item)
            for atoma in self.dstep[step]:
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatomid = [i for i in range(
                    len(step_atoms)) if distances[i] < self.cutoff]
                cutoffatoms = step_atoms[cutoffatomid]
                symbols = cutoffatoms.get_chemical_symbols()
                results.append((np.array([int(step), int(atoma)]), self._calcoulumbmatrix(
                    cutoffatoms), Counter(symbols)))
        return results

    def _calcoulumbmatrix(self, atoms):
        # https://github.com/crcollins/molml/blob/master/molml/utils.py
        top = np.outer(atoms.numbers, atoms.numbers).astype(np.float64)
        r = atoms.get_all_distances(mic=True)
        diag = np.array([self._coulumbdiag[symbol]
                         for symbol in atoms.get_chemical_symbols()])
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(top, r, top)
            np.fill_diagonal(top, diag)
        top[top == np.Infinity] = 0
        top[np.isnan(top)] = 0
        return np.linalg.eigh(top)[0]

    @classmethod
    def _clusterdatas(cls, X, n_clusters):
        min_max_scaler = preprocessing.MinMaxScaler()
        X = np.array(min_max_scaler.fit_transform(X))
        clus = MiniBatchKMeans(n_clusters=n_clusters, init_size=(
            3*n_clusters if 3*n_clusters < len(X) else len(X)))
        labels = clus.fit_predict(X)
        chooseindex = {}
        choosenum = {}
        for index, label in enumerate(labels):
            if label in chooseindex:
                r = np.random.randint(0, choosenum[label]+1)
                if r == 0:
                    chooseindex[label] = index
                choosenum[label] += 1
            else:
                chooseindex[label] = index
                choosenum[label] = 0
        index = np.array(list(chooseindex.values()))
        return index

    @classmethod
    def _mkdir(cls, path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    def _writexyzfiles(self):
        self.dstep = defaultdict(list)
        with open(os.path.join(self.trajatom_dir, "chooseatoms"), 'rb') as fc, open(self.dumpfilename) as f, open(self.bondfilename) as fb, Pool(self.nproc, maxtasksperchild=10000) as pool, tqdm(desc="Write structures", unit="structure", total=self._nstructure) as pbar:
            semaphore = Semaphore(360)
            for typefile, trajatomfilename in zip(fc, self.atombondtype):
                for line in self._decompress(typefile).split(";"):
                    s = line.split()
                    self.dstep[int(s[0])].append((int(s[1]), trajatomfilename))
            i = Counter()
            ii = 0
            maxlength = len(str(self.n_clusters))
            results = pool.imap_unordered(
                self._writestepxyzfile, self._produce(
                    semaphore,
                    enumerate(
                        zip(
                            itertools.islice(
                                itertools.zip_longest(
                                    *[f] * self.steplinenum),
                                0, None, self.stepinterval),
                            itertools.islice(
                                itertools.zip_longest(
                                    *[fb] * self.bondsteplinenum),
                                0, None, self.stepinterval))),
                    None),
                100)
            for result in results:
                for takenatoms, trajatomfilename in result:
                    pbar.update(1)
                    folder = str(ii//1000).zfill(3)
                    atomtypenum = str(i[trajatomfilename]).zfill(maxlength)
                    self._mkdir(os.path.join(self.dataset_dir, folder))
                    cutoffatoms = sum(takenatoms, Atoms())
                    cutoffatoms.wrap(
                        center=cutoffatoms[0].position /
                        takenatoms[0].get_cell_lengths_and_angles()[0: 3],
                        pbc=cutoffatoms.get_pbc())
                    write_xyz(
                        os.path.join(
                            self.dataset_dir, folder,
                            f'{self.xyzfilename}_{trajatomfilename}_{atomtypenum}.xyz'),
                        cutoffatoms, format='xyz')
                    if self.writegjf:
                        self._mkdir(os.path.join(self.gjfdir, folder))
                        self._convertgjf(
                            os.path.join(
                                self.gjfdir, folder,
                                f'{self.xyzfilename}_{trajatomfilename}_{atomtypenum}.gjf'),
                            takenatoms)
                    i[trajatomfilename] += 1
                    ii += 1
                semaphore.release()
            pool.close()
            pool.join()

    def _convertgjf(self, gjffilename, selected_atoms):
        buff = []
        # only support CHO, multiplicity of oxygen is 3
        multiplicities = list(
            (3
             if atoms.get_chemical_symbols() == ["O", "O"]
             else(Counter(atoms.get_chemical_symbols())['H'] % 2 + 1))
            for atoms in selected_atoms)
        atoms_whole = sum(selected_atoms, Atoms())
        atoms_whole.set_cell(selected_atoms[0].get_cell())
        atoms_whole.set_pbc(selected_atoms[0].get_pbc())
        multiplicity_whole = sum(multiplicities)-len(selected_atoms)+1
        title = '\nGenerated by MDDatasetMaker\n'
        if len(selected_atoms) == 1 or not self.fragment:
            atoms_whole.wrap(
                center=atoms_whole[0].position/atoms_whole.get_cell_lengths_and_angles()[0:3], pbc=atoms_whole.get_pbc())
            buff.extend((self.qmkeywords, title))
            buff.append(f'0 {multiplicity_whole}')
            buff.extend(("{} {:.5f} {:.5f} {:.5f}".format(
                atom.symbol, *atom.position)
                for atom in atoms_whole))
            buff.append('\n')
        else:
            for atoms in selected_atoms:
                atoms.wrap(center=atoms_whole[0].position/atoms_whole.get_cell_lengths_and_angles()[
                           0:3], pbc=atoms_whole.get_pbc())
            chk = f'%chk={os.path.splitext(os.path.basename(gjffilename))[0]}.chk'
            connect = '\n--link1--\n'
            kw1 = f'{self.qmkeywords} guess=fragment={len(selected_atoms)}'
            kw2 = f'{self.qmkeywords} force geom=chk guess=read'
            multiplicities_str = ' '.join(
                (f'0 {multiplicity}'
                 for multiplicity in itertools.chain(
                     (multiplicity_whole,),
                     multiplicities)))
            buff.extend((chk, kw1, title, multiplicities_str))
            for index, atoms in enumerate(selected_atoms, 1):
                buff.extend(('{}(Fragment={}) {:.5f} {:.5f} {:.5f}'.format(
                    atom.symbol, index, *atom.position) for atom in atoms))
            buff.extend((connect, chk, kw2,
                         title, multiplicities_str, '\n'))
        with open(gjffilename, 'w') as f:
            f.write('\n'.join(buff))

    def _writestepxyzfile(self, item):
        (step, (dumplines, bondlines)), _ = item
        results = []
        if step in self.dstep:
            step_atoms = self._readlammpscrdstep(((step, dumplines), None))
            molecules = self._readlammpsbondstepmolecules(bondlines)
            for atoma, trajatomfilename in self.dstep[step]:
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatomid = [i for i in range(
                    len(step_atoms)) if distances[i] < self.cutoff]
                # make cutoff atoms in molecules
                takenatomids = []
                for mo in molecules:
                    mol_atomid = [int(x)-1 for x in mo]
                    for moatom in mol_atomid:
                        if moatom in cutoffatomid:
                            takenatomids.append(mol_atomid)
                            break
                takenatoms = [step_atoms[takenatomid]
                              for takenatomid in takenatomids]
                results.append((takenatoms, trajatomfilename))
        return results

    def _readlammpsbondN(self, f):
        # copy from reacnetgenerator on 2018-12-15
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
        return steplinenum

    def _readlammpscrdN(self):
        # copy from reacnetgenerator on 2018-12-15
        with open(self.dumpfilename) as f:
            iscompleted = False
            for index, line in enumerate(f):
                if line.startswith("ITEM:"):
                    linecontent = 4 if line.startswith("ITEM: TIMESTEP") else (3 if line.startswith(
                        "ITEM: ATOMS") else (1 if line.startswith("ITEM: NUMBER OF ATOMS") else 2))
                else:
                    if linecontent == 1:
                        if iscompleted:
                            stepbindex = index
                            break
                        else:
                            iscompleted = True
                            stepaindex = index
        steplinenum = stepbindex-stepaindex
        return steplinenum

    def _readlammpsbondstepmolecules(self, lines):
        # copy from reacnetgenerator on 2018-12-15
        bond = [None for x in range(self._N)]
        for line in lines:
            if line:
                if not line.startswith("#"):
                    s = line.split()
                    bond[int(s[0])-1] = [int(x) for x in s[3:3+int(s[2])]]
        molecules = self._connectmolecule(bond)
        return molecules

    def _connectmolecule(self, bond):
        # copy from reacnetgenerator on 2018-12-15
        molecules = []
        done = np.zeros(self._N, dtype=bool)
        for i in range(1, self._N+1):
            if not done[i-1]:
                mole, done = self._mo(i, bond, [], done)
                molecules.append(sorted(mole))
        return molecules

    def _mo(self, i, bond, molecule, done):
        # copy from reacnetgenerator on 2018-12-15
        molecule.append(i)
        done[i-1] = True
        for b in bond[i-1]:
            if not done[b-1]:
                molecule, done = self._mo(b, bond, molecule, done)
        return molecule, done

    def _readlammpsbondstep(self, item):
        # copy from reacnetgenerator on 2018-12-15
        (step, lines), _ = item
        d = defaultdict(list)
        for line in lines:
            if line:
                if not line.startswith("#"):
                    s = line.split()
                    atombondstr = "".join(
                        str(x)
                        for x in sorted(
                            [max(1, round(float(x)))
                             for x in s[4 + int(s[2]): 4 + 2 * int(s[2])]]))
                    d[self.atomname[self.atomtype[int(
                        s[0])-1]-1]+atombondstr].append(int(s[0]))
        return d, step

    def _readtimestepsbond(self):
        # added on 2018-12-15
        stepatomfiles = {}
        self._mkdir(self.trajatom_dir)
        with open(self.bondfilename) as f, Pool(self.nproc, maxtasksperchild=10000) as pool:
            self.bondsteplinenum = self._readlammpsbondN(f)
            f.seek(0)
            semaphore = Semaphore(360)
            results = pool.imap_unordered(
                self._readlammpsbondstep, self._produce(
                    semaphore,
                    enumerate(
                        itertools.islice(
                            itertools.zip_longest(
                                *[f] * self.bondsteplinenum),
                            0, None, self.stepinterval)),
                    None),
                100)
            nstep = 0
            for d, step in tqdm(
                    results, desc="Read trajectory", unit="timestep"):
                for bondtype, atomids in d.items():
                    if bondtype not in self.atombondtype:
                        self.atombondtype.append(bondtype)
                        stepatomfiles[bondtype] = open(os.path.join(
                            self.trajatom_dir, f'stepatom.{bondtype}'), 'wb')
                    stepatomfiles[bondtype].write(self._compress(
                        ''.join((str(step), ' ', ','.join((str(x) for x in atomids)), '\n'))))
                semaphore.release()
                nstep += 1
        pool.close()
        self._nstep = nstep
        for stepatomfile in stepatomfiles.values():
            stepatomfile.close()
        pool.join()

    @classmethod
    def _compress(cls, x):
        return base64.a85encode(zlib.compress(x.encode()))+b'\n'

    @classmethod
    def _decompress(cls, x):
        return zlib.decompress(base64.a85decode(x.strip())).decode()


def _commandline():
    parser = argparse.ArgumentParser(description='MDDatasetBuilder')
    parser.add_argument('-d', '--dumpfile',
                        help='Input dump file, e.g. dump.reaxc', required=True)
    parser.add_argument(
        '-b', '--bondfile', help='Input bond file, e.g. bonds.reaxc',
        required=True)
    parser.add_argument('-a', '--atomname',
                        help='Atomic names in the trajectory, e.g. C H O',
                        nargs='*', required=True)
    parser.add_argument(
        '-np', '--nproc', help='Number of processes', type=int)
    parser.add_argument(
        '-c', '--cutoff', help='Cutoff radius (default is 5.0)', type=float,
        default=5.)
    parser.add_argument(
        '-i', '--interval', help='Step interval (default is 1)', type=int,
        default=1)
    parser.add_argument(
        '-s', '--size', help='Dataset size (default is 10,000)', type=int,
        default=10000)
    parser.add_argument(
        '-k', '--qmkeywords',
        help='QM keywords (default is %%nproc=4 #mn15/6-31g**)',
        default="%nproc=4\n#mn15/6-31g**")
    parser.add_argument(
        '-n', '--name', help='Dataset name (default is md)', default="md")
    args = parser.parse_args()
    DatasetBuilder(
        atomname=args.atomname, bondfilename=args.bondfile,
        dumpfilename=args.dumpfile, dataset_name=args.name, cutoff=args.cutoff,
        stepinterval=args.interval, n_clusters=args.size,
        qmkeywords=args.qmkeywords, nproc=args.nproc).builddataset()
