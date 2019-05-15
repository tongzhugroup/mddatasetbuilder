"""Gaussian logs to DeePMD data files."""

import argparse
import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.units import Ang, Bohr, Hartree, eV
from tqdm import tqdm

from gaussianrunner import GaussianAnalyst


class PrepareDeePMD:
    """Prepare DeePMD training files."""

    def __init__(
            self, data_path, atomname, deepmd_dir="data",
            jsonfilename=os.path.join("train", "train.json"),
            lattice="100 0 0 0 100 0 0 0 100", virial="0 0 0 0 0 0 0 0 0",
            nologs=False):
        """Init the class."""
        self.data_path = data_path
        self.atomname = atomname
        self.deepmd_dir = deepmd_dir
        self.system_paths = []
        self.batch_size = []
        self.lattcie = f"{lattice}\n"
        self.virial = f"{virial}\n"
        self.set_prefix = "set"
        self.setdir = f"{self.set_prefix}.000"
        self.jsonfilename = jsonfilename
        self.nologs = nologs

    def praparedeepmd(self):
        """Prepare the dataset."""
        self._searchpath()
        self._raw2np()
        self._writejson()

    def _searchpath(self):
        logfiles = []
        for root, _, files in tqdm(os.walk(self.data_path)):
            for logfile in files:
                if logfile.endswith(".out" if self.nologs else ".log"):
                    logfiles.append(os.path.join(root, logfile))
        with Pool() as pool:
            for result in pool.imap_unordered(self._preparedeepmdforLOG, tqdm(logfiles)):
                if result:
                    self._handleLOG(result)

    def _preparedeepmdforLOG(self, logfilename):
        if self.nologs:
            with open(logfilename, 'rb') as f:
                read_properties = pickle.load(f)
        else:
            read_properties = GaussianAnalyst(properties=[
                'energy', 'atomic_number', 'coordinate', 'force']).readFromLOG(logfilename)
        energy = read_properties['energy']
        atomic_number = read_properties['atomic_number']
        coord = read_properties['coordinate']
        force = read_properties['force']
        if energy is not None and atomic_number is not None and coord is not None and force is not None:
            energy *= Hartree/eV
            force *= (Hartree/Bohr)/(eV/Ang)
            id_sorted = np.argsort(atomic_number)
            n_ele = Counter(atomic_number)
            name = "".join(
                [f"{symbol}{n_ele[atomic_numbers[symbol]]}"
                 for symbol in self.atomname])
            path = os.path.join(self.deepmd_dir, f"data_{name}")
            return energy, atomic_number, coord, force, id_sorted, path
        else:
            return None

    def _handleLOG(self, result):
        energy, atomic_number, coord, force, id_sorted, path = result
        if not os.path.exists(path):
            os.makedirs(path)
            self.system_paths.append(path)
            self.batch_size.append(max(32//atomic_number.size, 1))
            with open(os.path.join(path, "type.raw"), 'w') as typefile:
                typefile.write(
                    " ".join(
                        (str(self.atomname.index(chemical_symbols[x]))
                         for x in np.sort(atomic_number))))
        with open(os.path.join(path, "coord.raw"), 'a') as coordfile, open(os.path.join(path, "force.raw"), 'a') as forcefile, open(os.path.join(path, "energy.raw"), 'a') as energyfile, open(os.path.join(path, "box.raw"), 'a') as boxfile, open(os.path.join(path, "virial.raw"), 'a') as virialfile:
            coordfile.write(
                f"{' '.join((str(x) for x in coord[id_sorted].flatten()))}\n")
            forcefile.write(
                f"{' '.join((str(x) for x in force[id_sorted].flatten()))}\n")
            energyfile.write(f"{energy}\n")
            boxfile.write(self.lattcie)
            virialfile.write(self.virial)

    def _raw2np(self):
        for i, system_path in enumerate(self.system_paths):
            if not os.path.exists(os.path.join(system_path, self.setdir)):
                os.makedirs(os.path.join(system_path, self.setdir))
            for dataname in ["box", "coord", "energy", "force", "virial"]:
                if os.path.isfile(
                        os.path.join(system_path, f"{dataname}.raw")):
                    data = np.loadtxt(os.path.join(
                        system_path, f"{dataname}.raw"))
                    if data.ndim == 1 and not dataname == "energy":
                        data = np.array([data])
                    data = data.astype(np.float32)
                    if dataname == "coord" and data.shape[0] < self.batch_size[i]:
                        self.batch_size[i] = data.shape[0]
                    np.save(os.path.join(
                        system_path, self.setdir, dataname), data)

    def _writejson(self):
        jsonpath = os.path.split(self.jsonfilename)[0]
        sel_a = [{"C": 40, "H": 80, "O": 40}.get(
            symbol, 40) for symbol in self.atomname]
        deepmd_json = {
            "use_smooth":       True,
            "sel_a":            sel_a,
            "rcut_smth":        1.,
            "rcut":             6.,
            "filter_neuron":    [25, 50, 100],
            "filter_resnet_dt": False,
            "axis_neuron":      12,
            "fitting_neuron":   [240, 240, 240],
            "fitting_resnet_dt": True,
            "coord_norm":       True,
            "type_fitting_net": False,
            "systems":          [os.path.relpath(path, jsonpath) for path in self.system_paths],
            "set_prefix":       self.set_prefix,
            "stop_batch":       4000000,
            "batch_size":       self.batch_size,
            "start_lr":         0.0005,
            "decay_steps":      20000,
            "decay_rate":       0.96,
            "start_pref_e":     0.2,
            "limit_pref_e":     0.2,
            "start_pref_f":     1000,
            "limit_pref_f":     1,
            "start_pref_v":     0,
            "limit_pref_v":     0,
            "seed":             1,
            "disp_file":        "lcurve.out",
            "disp_freq":        1000,
            "numb_test":        1,
            "save_freq":        1000,
            "save_ckpt":        "./model.ckpt",
            "load_ckpt":        "./model.ckpt",
            "disp_training":    True,
            "time_training":    True

        }
        if not os.path.exists(jsonpath):
            os.makedirs(jsonpath)
        with open(self.jsonfilename, 'w') as f:
            json.dump(deepmd_json, f)


def _commandline():
    parser = argparse.ArgumentParser(description='Prepare DeePMD data')
    parser.add_argument('-d', '--dir',
                        help='Dataset dirs, default is data', default="data")
    parser.add_argument(
        '-p', '--path', help='Gaussian LOG file path, e.g. dataset_md_GJF',
        required=True)
    parser.add_argument('-a', '--atomname',
                        help='Atomic names in the trajectory, e.g. C H O',
                        nargs='*', required=True)
    parser.add_argument(
        '--nologs', help='Read out files instead of logs', action="store_true")
    args = parser.parse_args()
    PrepareDeePMD(data_path=args.path, deepmd_dir=args.dir,
                  atomname=args.atomname, nologs=args.nologs).praparedeepmd()
