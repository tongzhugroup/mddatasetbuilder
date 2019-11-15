"""Gaussian logs to DeePMD data files."""

import argparse
import json
import os
import random
from multiprocessing import Pool

from tqdm import tqdm
import dpdata
import numpy as np


class PrepareDeePMD:
    """Prepare DeePMD training files."""

    def __init__(
            self, data_path, deepmd_dir="data",
            jsonfilenumber=1,
            fmt="gaussian/log", suffix=".log"):
        """Init the class."""
        self.data_path = data_path
        self.deepmd_dir = deepmd_dir
        self.system_paths = []
        self.batch_size = []
        self.fmt = fmt
        self.suffix = suffix
        self.jsonfilenames = [os.path.join(
            f"train{i}", f"train{i}.json") for i in range(jsonfilenumber)]

    def preparedeepmd(self):
        """Prepare the dataset."""
        self._searchpath()
        for jsonfilename in self.jsonfilenames:
            self._writejson(jsonfilename)

    def _searchpath(self):
        logfiles = []
        for root, _, files in tqdm(os.walk(self.data_path, followlinks=True)):
            for logfile in files:
                if logfile.endswith(self.suffix):
                    logfiles.append(os.path.join(root, logfile))
        multi_systems = dpdata.MultiSystems()
        with Pool() as pool:
            for system in pool.imap_unordered(self._preparedeepmdforLOG, tqdm(logfiles)):
                multi_systems.append(system)
        multi_systems.to_deepmd_npy(self.deepmd_dir)
        for formula, system in multi_systems.systems.items():
            self.system_paths.append(os.path.join(self.deepmd_dir, formula))
            self.batch_size.append(
                min(max(32//(system["coords"].shape[1]), 1), system["coords"].shape[0]))
        self.atomname = multi_systems.atom_names

    def _preparedeepmdforLOG(self, logfilename):
        system = dpdata.LabeledSystem(logfilename, fmt=self.fmt)
        atom_pref_file = os.path.splitext(logfilename)[0] + ".atom_pref.npy"
        if os.path.exists(atom_pref_file):
            system.data["atom_pref"] = np.load(atom_pref_file)
        return system

    def _writejson(self, jsonfilename):
        jsonpath = os.path.dirname(jsonfilename)
        try:
            sel_a = [{"C": 40, "H": 80, "O": 40}.get(
                symbol, 40) for symbol in self.atomname]
        except KeyError:
            raise RuntimeError("Unsupported atom types.")
        deepmd_json = {
            "model": {
                "type_map":     self.atomname,
                "descriptor": {
                    "type":         "se_a",
                    "sel":          sel_a,
                    "rcut_smth":    1.,
                    "rcut":         6.,
                    "neuron":       [25, 50, 100],
                    "resnet_dt":    False,
                    "axis_neuron":  12,
                    "seed":         random.randint(0, 2**32),
                },
                "fitting_net": {
                    "neuron":       [240, 240, 240],
                    "resnet_dt":    True,
                    "seed":         random.randint(0, 2**32),
                }
            },
            "learning_rate": {
                "type":             "exp",
                "start_lr":         0.0005,
                "decay_steps":      20000,
                "decay_rate":       0.96,
            },
            "loss": {
                "start_pref_e":     0.2,
                "limit_pref_e":     0.2,
                "start_pref_f":     1000,
                "limit_pref_f":     1,
                "start_pref_v":     0,
                "limit_pref_v":     0,
            },
            "training": {
                "systems":          [os.path.relpath(path, jsonpath) for path in self.system_paths],
                "set_prefix":       "set",
                "stop_batch":       4000000,
                "batch_size":       self.batch_size,
                "seed":             random.randint(0, 2**32),
                "disp_file":        "lcurve.out",
                "disp_freq":        1000,
                "numb_test":        1,
                "save_freq":        1000,
                "save_ckpt":        "./model.ckpt",
                "load_ckpt":        "./model.ckpt",
                "disp_training":    True,
                "time_training":    True
            }
        }
        if not os.path.exists(jsonpath):
            os.makedirs(jsonpath)
        with open(jsonfilename, 'w') as f:
            json.dump(deepmd_json, f)


def _commandline():
    parser = argparse.ArgumentParser(description='Prepare DeePMD data')
    parser.add_argument('-d', '--dir',
                        help='Dataset dirs, default is data', default="data")
    parser.add_argument(
        '-p', '--path', help='Gaussian LOG file path, e.g. dataset_md_GJF',
        required=True)
    parser.add_argument('-n', '--number', type=int, default=1,
                        help="The number of train.json")
    args = parser.parse_args()
    PrepareDeePMD(data_path=args.path, deepmd_dir=args.dir,
                  jsonfilenumber=args.number
                  ).preparedeepmd()
