#MDDatasetMaker
#Author: Jinzhe Zeng
#Email: njzjz@qq.com 10154601140@stu.ecnu.edu.cn
import itertools
import numpy as np
from ReacNetGenerator import ReacNetGenerator

class DatasetMaker(object):
    def __init__(self,n_eachspecies=10,bondfilename="bonds.reaxc",dumpfilename="dump.ch4",moleculefilename=None,tempfilename=None,dataset_dir="dataset",xyzfilename="md"):
        print("MDDatasetMaker")
        print("Author: Jinzhe Zeng")
        print("Email: njzjz@qq.com 10154601140@stu.ecnu.edu.cn")
        self.dumpfilename=dumpfilename
        self.bondfilename=bondfilename
        self.moleculefilename=moleculefilename if moleculefilename else self.bondfilename+".moname"
        self.tempfilename=tempfilename if tempfilename else self.bondfilename+".temp2"
        self.dataset_dir=dataset_dir
        self.xyzfilename=xyzfilename
        self.n_eachspecies=n_eachspecies
        self.dataset={}
        self.ReacNetGenerator=ReacNetGenerator(runHMM=False,inputfilename=self.bondfilename,moleculefilename=self.moleculefilename,moleculetemp2filename=self.tempfilename)

    def makedataset(self,processtraj=True):
        if processtraj:
            self.ReacNetGenerator.step1()
            self.ReacNetGenerator.step2()
            if self.ReacNetGenerator.SMILES:
                self.ReacNetGenerator.printmoleculeSMILESname()
            else:
                self.ReacNetGenerator.printmoleculename()
        self.readmoname()
        self.writexyzfile()

    def readmoname(self):
        datasets={}
        with open(self.moleculefilename) as fm,open(self.tempfilename) as ft:
            for linem,linet in zip(fm,ft):
                sm=linem.split()
                st=linet.split()
                mname=sm[0]
                matoms=np.array([int(x)-1 for x in sm[1].split(",")])
                steps=np.array([int(x) for x in st[-1].split(",")])
                for step in steps:
                    if mname in datasets:
                        if datasets[mname][0]+1<self.n_eachspecies:
                            datasets[mname][1].append((step,matoms))
                        else:
                            r=np.random.randint(0,datasets[mname][0]+1)
                            if r<=self.n_eachspecies-1:
                                datasets[mname][1][r]=(step,matoms)
                        datasets[mname][0]+=1
                    else:
                        datasets[mname]=[0,[(step,matoms)]]
        for spec in datasets.values():
            for struc in spec[1]:
                if struc[0] in self.dataset:
                    self.dataset[struc[0]].append(struc[1])
                else:
                    self.dataset[struc[0]]=[struc[1]]

    def readlammpscrdstep(self,item):
        (step,lines),_=item
        atomtype=np.zeros((self.ReacNetGenerator.N),dtype=np.int)
        atomcrd=np.zeros((self.ReacNetGenerator.N,3))
        for line in lines:
            if line:
                if line.startswith("ITEM:"):
                    if line.startswith("ITEM: TIMESTEP"):
                        linecontent=4
                    elif line.startswith("ITEM: ATOMS"):
                        linecontent=3
                    elif line.startswith("ITEM: NUMBER OF ATOMS"):
                        linecontent=1
                    elif line.startswith("ITEM: BOX BOUNDS"):
                        linecontent=2
                else:
                    if linecontent==3:
                        s=line.split()
                        atomtype[int(s[0])-1]=int(s[1])
                        atomcrd[int(s[0])-1]=float(s[2]),float(s[3]),float(s[4])
        return atomtype,atomcrd

    def writexyzfile(self):
        i=0
        self.ReacNetGenerator.inputfilename=self.dumpfilename
        steplinenum=self.ReacNetGenerator.readlammpscrdN()
        with open(self.dumpfilename) as f:
            for item in enumerate(itertools.islice(itertools.zip_longest(*[f]*steplinenum),0,None,1)):
                if item[0] in self.dataset:
                    atomtype,atomcrd=self.readlammpscrdstep((item,None))
                    for struc in self.dataset[item[0]]:
                        self.ReacNetGenerator.convertxyz(atomtype[struc],atomcrd[struc],self.dataset_dir+"/"+self.xyzfilename+"_"+str(i)+".xyz")
                        i+=1

if __name__ == '__main__':
    DatasetMaker(bondfilename="bonds.reaxc.ch4",dataset_dir="dataset_ch4",xyzfilename="ch4").makedataset(processtraj=False)
