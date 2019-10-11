"""MDDatasetBuilder.

Run 'datasetbuilder -h' for more details.
"""

__author__ = "Jinzhe Zeng"
__email__ = "jzzeng@stu.ecnu.edu.cn"
__update__ = '2019-07-11'
__date__ = '2018-07-18'

import argparse
import gc
import itertools
import logging
import os
import tempfile
import time
import pickle
from collections import Counter, defaultdict
from multiprocessing import cpu_count

import numpy as np
from ase.data import atomic_numbers
from ase.io import write as write_xyz
from pkg_resources import DistributionNotFound, get_distribution
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans

from .detect import Detect
from .utils import run_mp, bytestolist, listtobytes, must_be_list

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = ''


class DatasetBuilder:
    """Dataset Builder."""

    def __init__(
            self, atomname=None,
            clusteratom=None, bondfilename=None,
            dumpfilename="dump.reaxc", dataset_name="md", cutoff=5,
            stepinterval=1, n_clusters=10000, n_each=1,
            qmkeywords="%nproc=4\n#mn15/6-31g(d,p) force", nproc=None, pbc=True,
            fragment=True, errorfilename=None, errorlimit=0.):
        """Init the builder."""
        print(__doc__)
        print(f"Author:{__author__}  Email:{__email__}")
        atomname = np.array(
            atomname) if atomname else np.array(["C", "H", "O"])
        self.crddetector = Detect.gettype('dump')(
            filename=dumpfilename, atomname=atomname, pbc=pbc, errorfilename=errorfilename, errorlimit=errorlimit)
        if bondfilename is None:
            self.bonddetector = self.crddetector
        else:
            self.bonddetector = Detect.gettype('bond')(
                filename=bondfilename, atomname=atomname, pbc=pbc)

        self.dataset_dir = f"dataset_{dataset_name}"
        self.xyzfilename = dataset_name
        self.clusteratom = clusteratom if clusteratom else atomname
        self.atombondtype = []
        self.stepinterval = stepinterval
        self.nproc = nproc if nproc else cpu_count()
        self.cutoff = cutoff
        self.n_clusters = n_clusters
        self.n_each = n_each
        self.writegjf = True
        self.gjfdir = f'{self.dataset_dir}_gjf'
        self.qmkeywords = must_be_list(qmkeywords)
        self.fragment = fragment
        self._coulumbdiag = dict(map(lambda symbol: (
            symbol, atomic_numbers[symbol]**2.4/2), atomname))
        self._nstructure = 0
        self.bondtyperestore = {}
        self.errorfilename = errorfilename

    def builddataset(self, writegjf=True):
        """Build a dataset."""
        self.writegjf = writegjf
        timearray = [time.time()]
        with tempfile.TemporaryDirectory() as self.trajatom_dir:
            for runstep in range(3):
                if runstep == 0:
                    self._readtimestepsbond()
                elif runstep == 1:
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

    def _readtimestepsbond(self):
        # added on 2018-12-15
        stepatomfiles = {}
        self._mkdir(self.trajatom_dir)
        results = run_mp(self.nproc, func=self.bonddetector.readatombondtype,
                         l=zip(self.lineiter(self.bonddetector), self.erroriter(
                         )) if self.errorfilename is not None else self.lineiter(self.bonddetector),
                         return_num=True, extra=self.errorfilename is not None,
                         desc="Read trajectory", unit="timestep")
        nstep = 0
        for d, step in results:
            for bondtypebytes, atomids in d.items():
                bondtype = self._bondtype(bondtypebytes)
                if bondtype not in self.atombondtype:
                    self.atombondtype.append(bondtype)
                    stepatomfiles[bondtype] = open(os.path.join(
                        self.trajatom_dir, f'stepatom.{bondtype}'), 'wb')
                stepatomfiles[bondtype].write(
                    listtobytes([step, atomids]))
            nstep += 1
        self._nstep = nstep
        for stepatomfile in stepatomfiles.values():
            stepatomfile.close()

    def _writecoulumbmatrix(self, trajatomfilename, fc):
        self.dstep = {}
        with open(os.path.join(self.trajatom_dir, f"stepatom.{trajatomfilename}"), 'rb') as f:
            for line in f:
                s = bytestolist(line)
                self.dstep[s[0]] = s[1]
        n_atoms = sum(map(len, self.dstep.values()))
        if n_atoms > self.n_clusters:
            # undersampling
            max_counter = Counter()
            stepatom = np.zeros((n_atoms, 2), dtype=int)
            feedvector = np.zeros((n_atoms, 0))
            vector_elements = defaultdict(list)
            results = run_mp(self.nproc, func=self._writestepmatrix,
                             l=self.lineiter(self.crddetector),
                             return_num=True, total=self._nstep,
                             desc=trajatomfilename, unit="timestep")
            j = 0
            for result in results:
                for stepatoma, vector, symbols_counter in result:
                    stepatom[j] = stepatoma
                    for element in (
                            symbols_counter - max_counter).elements():
                        vector_elements[element].append(
                            feedvector.shape[1])
                        feedvector = np.pad(
                            feedvector, ((0, 0), (0, 1)),
                            'constant',
                            constant_values=(0, self._coulumbdiag
                                             [element]))
                    feedvector[j, sum(map(
                        lambda x:vector_elements[x[0]][: x[1]], symbols_counter.items()), [])]=vector
                    max_counter |= symbols_counter
                    j += 1
            logging.info(
                f"Max counter of {trajatomfilename} is {max_counter}")
            choosedindexs = self._clusterdatas(
                np.sort(feedvector), n_clusters=self.n_clusters,
                n_each=self.n_each)
        else:
            stepatom = np.array([[u, vv]
                                 for u, v in self.dstep.items() for vv in v])
            choosedindexs = range(n_atoms)
        fc.write(listtobytes(stepatom[choosedindexs]))
        self._nstructure += len(choosedindexs)

    def _writestepmatrix(self, item):
        step, lines = item
        results = []
        if step in self.dstep:
            step_atoms, _ = self.crddetector.readcrd(lines)
            for atoma in self.dstep[step]:
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatoms = step_atoms[distances < self.cutoff]
                symbols = cutoffatoms.get_chemical_symbols()
                results.append(
                    (np.array([step, atoma]),
                     self._calcoulumbmatrix(cutoffatoms),
                     Counter(symbols)))
        return results

    def _calcoulumbmatrix(self, atoms):
        # https://github.com/crcollins/molml/blob/master/molml/utils.py
        top = np.outer(atoms.numbers, atoms.numbers).astype(np.float64)
        r = atoms.get_all_distances(mic=True)
        diag = np.array(
            list(map(self._coulumbdiag.get, atoms.get_chemical_symbols())))
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(top, r, top)
            np.fill_diagonal(top, diag)
        top[top == np.Infinity] = 0
        top[np.isnan(top)] = 0
        return np.linalg.eigh(top)[0]

    @classmethod
    def _clusterdatas(cls, X, n_clusters, n_each=1):
        min_max_scaler = preprocessing.MinMaxScaler()
        X = np.array(min_max_scaler.fit_transform(X))
        clus = MiniBatchKMeans(n_clusters=n_clusters, init_size=(
            min(3*n_clusters, len(X))))
        labels = clus.fit_predict(X)
        choosedidx = []
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if idx.size:
                choosedidx.append(np.random.choice(idx, n_each))
        index = np.concatenate(choosedidx)
        return index

    @classmethod
    def _mkdir(cls, path):
        try:
            os.makedirs(path)
        except OSError:
            pass

    def _writexyzfiles(self):
        self.dstep = defaultdict(list)
        with open(os.path.join(self.trajatom_dir, "chooseatoms"), 'rb') as fc:
            typecounter = Counter()
            for typefile, trajatomfilename in zip(fc, self.atombondtype):
                for step, atoma in bytestolist(typefile):
                    self.dstep[step].append(
                        (atoma, trajatomfilename,
                         typecounter[trajatomfilename],
                         typecounter['total']))
                    typecounter[trajatomfilename] += 1
                    typecounter['total'] += 1
            self.maxlength = len(str(self.n_clusters))
            foldernum = self._nstructure//1000 + 1
            self.foldermaxlength = len(str(foldernum))
            foldernames = list(map(lambda i: str(i).zfill(
                self.foldermaxlength), range(foldernum)))
            for folder in foldernames:
                self._mkdir(os.path.join(self.dataset_dir, folder))
            if self.writegjf:
                for folder in foldernames:
                    self._mkdir(os.path.join(self.gjfdir, folder))
            crditer = self.lineiter(self.crddetector)
            if self.crddetector is self.bonddetector:
                lineiter = crditer
            else:
                bonditer = self.lineiter(self.bonddetector)
                lineiter = zip(crditer, bonditer)
            results = run_mp(self.nproc, func=self._writestepxyzfile,
                             l=lineiter, return_num=True,
                             total=self._nstructure,
                             desc="Write structures", unit="structure")
            for _ in results:
                pass

    @staticmethod
    def detect_multiplicity(symbols):
        # currently only support charge=0
        # oxygen -> 3
        if np.count_nonzero(symbols == ["O"]) == 2 and symbols.size == 2:
            return 3
        # calculates the total number of electrons, assumes they are paired as much as possible
        n_total = sum([atomic_numbers[s] for s in symbols])
        return n_total % 2 + 1

    def _convertgjf(self, gjffilename, takenatomidindex, atoms_whole):
        buff = []
        multiplicities = list(map(lambda atoms: self.detect_multiplicity(
            atoms_whole[atoms].get_chemical_symbols()), takenatomidindex))
        multiplicity_whole = sum(multiplicities)-len(takenatomidindex)+1
        multiplicity_whole_str = f'0 {multiplicity_whole}'
        title = '\nGenerated by MDDatasetMaker (Author: Jinzhe Zeng)\n'
        if len(self.qmkeywords) > 1:
            connect = '\n--link1--\n'
            chk = [
                f'%chk={os.path.splitext(os.path.basename(gjffilename))[0]}.chk']
        else:
            chk = []
        if len(takenatomidindex) == 1 or not self.fragment:
            buff.extend(
                (*chk, self.qmkeywords[0], title, multiplicity_whole_str))
            buff.extend(map(lambda atom: "{} {:.5f} {:.5f} {:.5f}".format(
                atom.symbol, *atom.position), atoms_whole))
        else:
            kw0 = f'{self.qmkeywords[0]} guess=fragment={len(takenatomidindex)}'
            multiplicities_str = "{} {}".format(multiplicity_whole_str, ' '.join(
                [f'0 {multiplicity}' for multiplicity in multiplicities]))
            buff.extend((*chk, kw0, title, multiplicities_str))
            for index, atoms in enumerate(takenatomidindex, 1):
                buff.extend(map(lambda atom: '{}(Fragment={}) {:.5f} {:.5f} {:.5f}'.format(
                    atom.symbol, index, *atom.position), atoms_whole[atoms]))
        for kw in itertools.islice(self.qmkeywords, 1, None):
            buff.extend((connect, *chk, kw,
                         title, f'0 {multiplicity_whole}', '\n'))
        buff.append('\n')
        with open(gjffilename, 'w') as f:
            f.write('\n'.join(buff))

    def _writestepxyzfile(self, item):
        step, lines = item
        results = 0
        if step in self.dstep:
            if len(lines) == 2:
                step_atoms, _ = self.crddetector.readcrd(lines[0])
                molecules = self.bonddetector.readmolecule(lines[1])
            else:
                molecules, step_atoms = self.bonddetector.readmolecule(lines)
            for atoma, trajatomfilename, itype, itotal in self.dstep[step]:
                # update counter
                folder = str(itotal//1000).zfill(self.foldermaxlength)
                atomtypenum = str(itype).zfill(self.maxlength)
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatomid = np.where(distances < self.cutoff)
                # make cutoff atoms in molecules
                takenatomids = []
                takenatomidindex = []
                idsum = 0
                for mo in molecules:
                    mol_atomid = np.array(mo)
                    if np.any(np.isin(mol_atomid, cutoffatomid)):
                        takenatomids.append(mol_atomid)
                        takenatomidindex.append(
                            range(idsum, idsum+len(mol_atomid)))
                        idsum += len(mol_atomid)
                idx = np.concatenate(takenatomids)
                cutoffatoms = step_atoms[idx]
                cutoffatoms[np.nonzero(idx == atoma-1)[0][0]].tag = 1
                cutoffatoms.wrap(
                    center=step_atoms[atoma-1].position /
                    cutoffatoms.get_cell_lengths_and_angles()[0: 3],
                    pbc=cutoffatoms.get_pbc())
                write_xyz(
                    os.path.join(
                        self.dataset_dir, folder,
                        f'{self.xyzfilename}_{trajatomfilename}_{atomtypenum}.xyz'),
                    cutoffatoms, format='xyz')
                if self.writegjf:
                    self._convertgjf(
                        os.path.join(
                            self.gjfdir, folder,
                            f'{self.xyzfilename}_{trajatomfilename}_{atomtypenum}.gjf'),
                        takenatomidindex, cutoffatoms)
                    np.save(os.path.join(
                        self.gjfdir, folder,
                        f'{self.xyzfilename}_{trajatomfilename}_{atomtypenum}.atom_pref.npy'),
                        np.array([cutoffatoms.get_tags()]))
                results += 1
        return results

    def _bondtype(self, typebytes):
        if typebytes in self.bondtyperestore:
            return self.bondtyperestore[typebytes]
        typetuple = pickle.loads(typebytes)
        typestr = f"{typetuple[0]}{''.join(map(str,typetuple[1]))}"
        self.bondtyperestore[typebytes] = typestr
        return typestr

    def lineiter(self, detector):
        fns = must_be_list(detector.filename)
        for fn in fns:
            with open(fn) as f:
                it = itertools.islice(itertools.zip_longest(
                    *[f] * detector.steplinenum), 0, None, self.stepinterval)
                for line in it:
                    yield line

    def erroriter(self):
        fns = must_be_list(self.errorfilename)
        for fn in fns:
            with open(fn) as f:
                it = itertools.islice(f, 1, None)
                for line in it:
                    yield line


def _commandline():
    parser = argparse.ArgumentParser(description='MDDatasetBuilder')
    parser.add_argument('-d', '--dumpfile', nargs='*',
                        help='Input dump file, e.g. dump.reaxc', required=True)
    parser.add_argument(
        '-b', '--bondfile', nargs='*', help='Input bond file, e.g. bonds.reaxc')
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
    parser.add_argument(
        '--errorfile', help='Error file generated by modified DeePMD', nargs='*')
    parser.add_argument(
        '-e', '--errorlimit', help='Error Limit', type=float, default=0.)
    args = parser.parse_args()
    DatasetBuilder(
        atomname=args.atomname, bondfilename=args.bondfile,
        dumpfilename=args.dumpfile, dataset_name=args.name, cutoff=args.cutoff,
        stepinterval=args.interval, n_clusters=args.size,
        qmkeywords=args.qmkeywords, nproc=args.nproc,
        errorfilename=args.errorfile, errorlimit=args.errorlimit
    ).builddataset()
