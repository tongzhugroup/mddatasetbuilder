import os
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers
from gaussianrunner import GaussianAnalyst

atomname=["C","H","O"]

def makedeepmd(logfilename, dir="deepmd"):
    read_properties = GaussianAnalyst(properties=[
                                      'energy', 'atomic_number', 'coordinate', 'force']).readFromLOG(logfilename)
    energy, atomic_number, coord, force = read_properties['energy'], read_properties[
        'atomic_number'], read_properties['coordinate'], read_properties['force']
    if energy is not None and atomic_number is not None and coord is not None and force is not None:
        

        # todo
        id_sorted, n_atom_sorted = list(
            zip(*sorted(n_atom.items(), key=lambda x: x[1]))) 


        n_ele = Counter(atomic_number)

        # todo
        name = "".join([chemical_symbols[x]+str(n_ele[x]) for x in [1, 8]])

        # todo
        path = dir+"/data_"+name
        
        if not os.path.exists(path):
            os.makedirs(path)
            with open(path+"/type.raw", 'w') as typefile:
                typefile.write(" ".join([str(atomname.index([atomic_numbers[x]))
                                for x in n_atom_sorted]))
        with open(path+"/coord.raw", 'a') as coordfile, open(path+"/force.raw", 'a') as forcefile, open(path+"/energy.raw", 'a') as energyfile, open(path+"/box.raw", 'a') as boxfile, open(path+"/virial.raw", 'a') as virialfile:
            for atomid in id_sorted:
                print(crd[atomid][0], crd[atomid][1],
                    crd[atomid][2], end=' ', file=coordfile)
                print(force[atomid][0], force[atomid][1],
                    force[atomid][2], end=' ', file=forcefile)
            print('', file=coordfile)
            print('', file=forcefile)
            print(energy, file=energyfile)
            print(lattcie, file=boxfile)
            print("0 0 0 0 0 0 0 0 0", file=virialfile)
