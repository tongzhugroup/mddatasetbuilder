from reacnetgenerator import ReacNetGenerator
from collections import Counter
from ase.data import atomic_numbers
from ase.io import write as write_xyz
from ase import Atoms, Atom
from multiprocessing import Pool, Semaphore, cpu_count
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import time
import shutil
import gc
import os
import numpy as np
import itertools
''' MDDatasetMaker '''
__author__ = "Jinzhe Zeng"
__email__ = "jzzeng@stu.ecnu.edu.cn"


class DatasetMaker(object):
    def __init__(self, atomname=["C", "H", "O"], clusteratom=["C", "H", "O"], bondfilename="bonds.reaxc", dumpfilename="dump.ch4", moleculefilename=None, tempfilename=None, dataset_dir="dataset", xyzfilename="md", cutoff=5, stepinterval=1, n_clusters=10000, qmkeywords="%nproc=4\n#force mn15/6-31g(d,p)", nproc=None, pbc=True):
        print(__doc__)
        print("Author:{}  Email:{}".format(__author__, __email__))
        self.dumpfilename = dumpfilename
        self.bondfilename = bondfilename
        self.moleculefilename = moleculefilename if moleculefilename else self.bondfilename+".moname"
        self.tempfilename = tempfilename if tempfilename else self.bondfilename+".temp2"
        self.dataset_dir = dataset_dir
        self.xyzfilename = xyzfilename
        self.atomname = np.array(atomname)
        self.clusteratom = clusteratom
        self.atombondtype = []
        self.trajatom_dir = "trajatom"
        self.stepinterval = stepinterval
        self.nproc = nproc if nproc else cpu_count()
        self.ReacNetGenerator = ReacNetGenerator(atomname=self.atomname, runHMM=False, inputfilename=self.bondfilename,
                                                 moleculefilename=self.moleculefilename, moleculetemp2filename=self.tempfilename, stepinterval=self.stepinterval, nproc=self.nproc)
        self.cutoff = cutoff
        self.n_clusters = n_clusters
        self.writegjf = True
        self.gjfdir = '{}_gjf'.format(self.dataset_dir)
        self.qmkeywords = qmkeywords
        self.pbc = pbc

    def logging(self, *message):
        localtime = time.asctime(time.localtime(time.time()))
        print(localtime, 'MDDatasetMaker:', *message)

    def produce(self, semaphore, list, parameter):
        for item in list:
            semaphore.acquire()
            yield item, parameter

    def makedataset(self, processtraj=True, writegjf=True):
        self.writegjf = writegjf
        timearray = self.printtime([])
        for runstep in range(6):
            if runstep == 0:
                if processtraj:
                    self.logging(
                        "Analyze the trajectory via ReacNetGenerator......")
                    self.ReacNetGenerator.inputfilename = self.bondfilename
                    self.ReacNetGenerator.run()
            elif runstep == 1:
                self.readlammpscrdN()
                self.readmoname()
            elif runstep == 2:
                self.processatomtypelist(self.sorttrajatom)
            elif runstep == 3:
                self.processatomtypelist(self.writecoulumbmatrix)
            elif runstep == 4:
                self.processatomtypelist(self.selectatoms)
            elif runstep == 5:
                self.mkdir(self.dataset_dir)
                if self.writegjf:
                    self.mkdir(self.gjfdir)
                self.processatomtypelist(self.writexyzfile)
            gc.collect()
            timearray = self.printtime(timearray)

    def printtime(self, timearray):
        timearray.append(time.time())
        if len(timearray) > 1:
            self.logging("Step ", len(timearray)-1, " has been completed. Time consumed: ",
                         round(timearray[-1]-timearray[-2], 3), "s")
        return timearray

    def readmoname(self):
        self.mkdir(self.trajatom_dir)
        with open(self.moleculefilename) as fm, open(self.tempfilename, 'rb') as ft:
            for linem, linet in zip(fm, ft):
                sm = linem.split()
                st = self.ReacNetGenerator._decompress(linet).split()
                matoms = np.array([int(x) for x in sm[1].split(",")])
                mbonds = np.array([[int(y) for y in x.split(",")] for x in sm[2].split(
                    ";")]) if len(sm) == 3 else np.array([])
                for atom in matoms:
                    atomname = self.atomname[self.atomtype[atom-1]-1]
                    if atomname in self.clusteratom:
                        atombond = [
                            bond[2] for bond in mbonds if bond[0] == atom or bond[1] == atom]
                        atombondstr = "".join(str(x) for x in sorted(atombond))
                        bondtype = atomname+atombondstr
                        with open(os.path.join(self.trajatom_dir, "trajatom."+bondtype), 'a' if bondtype in self.atombondtype else 'w') as f:
                            print(atom, st[-1], file=f)
                        if not bondtype in self.atombondtype:
                            self.atombondtype.append(bondtype)
        with open("trajatom.list", "w") as f:
            for atombondstr in self.atombondtype:
                print(atombondstr, file=f)

    def sorttrajatom(self, trajatomfilename):
        dstep = {}
        with open(os.path.join(self.trajatom_dir, "trajatom."+trajatomfilename)) as f:
            for line in f:
                s = line.split()
                steps = np.array([int(x) for x in s[-1].split(",")])
                for step in steps:
                    if step in dstep:
                        dstep[step].append(s[0])
                    else:
                        dstep[step] = [s[0]]
        with open(os.path.join(self.trajatom_dir, "stepatom."+trajatomfilename), "w") as f:
            for step, atoms in dstep.items():
                print(step, ",".join(atoms), file=f)

    def trajlist(self):
        with open("trajatom.list") as f:
            for line in f:
                gc.collect()
                yield line.strip()

    def processatomtypelist(self, func):
        for line in self.trajlist():
            func(line)

    def readlammpscrdstep(self, item):
        (step, lines), _ = item
        boxsize = []
        step_atoms = []
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    if line.startswith("ITEM: TIMESTEP"):
                        linecontent = 4
                    elif line.startswith("ITEM: ATOMS"):
                        linecontent = 3
                    elif line.startswith("ITEM: NUMBER OF ATOMS"):
                        linecontent = 1
                    elif line.startswith("ITEM: BOX BOUNDS"):
                        linecontent = 2
                else:
                    if linecontent == 3:
                        s = line.split()
                        step_atoms.append(
                            (int(s[0]), Atom(self.atomname[int(s[1])-1], [float(x) for x in s[2:5]])))
                    elif linecontent == 2:
                        s = line.split()
                        boxsize.append(float(s[1])-float(s[0]))
        # sort by ID
        _, step_atoms = zip(*sorted(step_atoms, key=lambda a: a[0]))
        step_atoms = Atoms(step_atoms, cell=boxsize, pbc=self.pbc)
        return step_atoms

    def readlammpscrdN(self):
        self.ReacNetGenerator.inputfilename = self.bondfilename
        self.bondsteplinenum = self.ReacNetGenerator._readlammpsbondN()
        self.ReacNetGenerator.inputfilename = self.dumpfilename
        self.steplinenum = self.ReacNetGenerator._readlammpscrdN()
        self.atomtype = self.ReacNetGenerator._atomtype

    def writecoulumbmatrix(self, trajatomfilename):
        self.dstep = {}
        with open(os.path.join(self.trajatom_dir, "stepatom."+trajatomfilename)) as f:
            for line in f:
                s = line.split()
                self.dstep[int(s[0])] = [int(x) for x in s[1].split(",")]
        with open(self.dumpfilename) as f, open(os.path.join(self.trajatom_dir, "coulumbmatrix."+trajatomfilename), 'w') as fm, Pool(self.nproc, maxtasksperchild=100) as pool:
            semaphore = Semaphore(360)
            results = pool.imap_unordered(self.writestepmatrix, self.produce(semaphore, enumerate(
                itertools.islice(itertools.zip_longest(*[f]*self.steplinenum), 0, None, self.stepinterval)), None), 10)
            for result in results:
                for resultline in result:
                    print(*resultline, file=fm)
                semaphore.release()

    def writestepmatrix(self, item):
        (step, lines), _ = item
        results = []
        if step in self.dstep:
            step_atoms = self.readlammpscrdstep(item)
            for atoma in self.dstep[step]:
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatomid = [i for i in range(
                    len(step_atoms)) if distances[i] < self.cutoff]
                cutoffatoms = step_atoms[cutoffatomid]
                results.append((str(step), str(atoma), ",".join(str(x) for x in self.calcoulumbmatrix(
                    cutoffatoms)), ",".join(cutoffatoms.get_chemical_symbols())))
        return results

    def calcoulumbmatrix(self, atoms):
        return -np.sort(-np.linalg.eig([[atoms[i].number**2.4/2 if i == j else atoms[i].number*atoms[j].number/atoms.get_distance(i, j, mic=True) for j in range(len(atoms))] for i in range(len(atoms))])[0])

    def selectatoms(self, trajatomfilename):
        coulumbmatrix = []
        stepatom = []
        max_counter = Counter()
        with open(os.path.join(self.trajatom_dir, "coulumbmatrix."+trajatomfilename)) as f:
            for line in f:
                s = line.split()
                stepatom.append([s[x] for x in range(2)])
                mline = np.array([float(x) for x in s[2].split(",")])
                symbols = s[3].split(",")
                max_counter |= Counter(symbols)
                coulumbmatrix.append((mline, symbols))
        self.logging("Max counter of", trajatomfilename, "is", max_counter)
        choosedindexs, choosedvalues = self.clusterdatas(np.array([np.concatenate([-np.sort(-np.concatenate((mline[[idx for idx, s in enumerate(symbols) if s == symbol]], [atomic_numbers[symbol]**2.4/2]*(
            max_counter[symbol]-symbols.count(symbol))))) for symbol in max_counter]) for mline, symbols in coulumbmatrix]), n_clusters=self.n_clusters, undersampling=(True if len(coulumbmatrix) > self.n_clusters else False))
        with open(os.path.join(self.trajatom_dir, "chooseatoms."+trajatomfilename), 'w') as f, open(os.path.join(self.trajatom_dir, "vector."+trajatomfilename), 'w') as fv:
            for index, value in zip(choosedindexs, choosedvalues):
                print(*stepatom[index], file=f)
                print(*value, file=fv)

    def clusterdatas(self, X, n_clusters, undersampling):
        min_max_scaler = preprocessing.MinMaxScaler()
        X = np.array(min_max_scaler.fit_transform(X))
        if undersampling:
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
        else:
            index = np.arange(len(X))
        return index, X[index]

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def writexyzfile(self, trajatomfilename):
        self.dstep = {}
        with open(os.path.join(self.trajatom_dir, "chooseatoms."+trajatomfilename)) as f:
            for line in f:
                s = line.split()
                if int(s[0]) in self.dstep:
                    self.dstep[int(s[0])].append(int(s[1]))
                else:
                    self.dstep[int(s[0])] = [int(s[1])]
        with open(self.dumpfilename) as f, open(self.bondfilename) as fb, Pool(self.nproc, maxtasksperchild=100) as pool:
            semaphore = Semaphore(360)
            i = 0
            results = pool.imap_unordered(self.writestepxyzfile, self.produce(semaphore, enumerate(zip(itertools.islice(itertools.zip_longest(
                *[f]*self.steplinenum), 0, None, self.stepinterval), itertools.islice(itertools.zip_longest(*[fb]*self.bondsteplinenum), 0, None, self.stepinterval))), None), 10)
            for result in results:
                for cutoffatoms, oxygennum in result:
                    cutoffatoms.wrap(
                        center=cutoffatoms[0].position/cutoffatoms.get_cell_lengths_and_angles()[0:3], pbc=cutoffatoms.get_pbc())
                    write_xyz(os.path.join(self.dataset_dir, '{}_{}_{}.xyz'.format(
                        self.xyzfilename, trajatomfilename, i)), cutoffatoms, format='xyz')
                    if self.writegjf:
                        self.convertgjf(os.path.join(
                            self.gjfdir, '{}_{}_{}.gjf'.format(self.xyzfilename, trajatomfilename, i)), cutoffatoms, oxygennum)
                    i += 1
                semaphore.release()

    def convertgjf(self, gjffilename, atoms, oxygennum):
        # only support CHO
        S = atoms.get_chemical_symbols().count("H") % 2+1+oxygennum*2
        buff = [self, qmkeywords, '', '{} n_oxygen={} by MDDatasetMaker'.format(
            gjffilename, oxygennum), '', '0 {}'.format(S)]
        buff.extend(('{} {} {} {}'.format(atom.symbol, *atom.position)
                     for atom in atoms))
        buff.append('')
        with open(gjffilename, 'w') as f:
            f.write('\n'.join(buff))

    def writestepxyzfile(self, item):
        (step, (dumplines, bondlines)), _ = item
        results = []
        if step in self.dstep:
            step_atoms = self.readlammpscrdstep(((step, dumplines), None))
            molecules = self.ReacNetGenerator._readlammpsbondstep(
                ((step, bondlines), None))[0]
            for atoma in self.dstep[step]:
                # atom ID starts from 1
                distances = step_atoms.get_distances(
                    atoma-1, range(len(step_atoms)), mic=True)
                cutoffatomid = [i for i in range(
                    len(step_atoms)) if distances[i] < self.cutoff]
                # make cutoff atoms in molecules
                oxygennum = 0
                for mo in molecules:
                    mol_atomid = [int(x)-1 for x in mo.split()[0].split(",")]
                    for moatom in mol_atomid:
                        if moatom in cutoffatomid:
                            cutoffatomid = list(
                                set(cutoffatomid) | set(mol_atomid))
                            # oxygen
                            if step_atoms[mol_atomid].get_chemical_symbols() == ['O', 'O']:
                                oxygennum += 1
                            break
                cutoffatoms = step_atoms[cutoffatomid]
                results.append((cutoffatoms, oxygennum))
        return results
