#MDDatasetMaker
#Author: Jinzhe Zeng
#Email: njzjz@qq.com 10154601140@stu.ecnu.edu.cn
import itertools
import numpy as np
import os
import gc
import shutil
import time
from sklearn.cluster import MiniBatchKMeans
from ReacNetGenerator import ReacNetGenerator
from multiprocessing import Pool, Semaphore

class DatasetMaker(object):
    def __init__(self,atomname=["C","H","O"],clusteratom=["C","H","O"],bondfilename="bonds.reaxc",dumpfilename="dump.ch4",moleculefilename=None,tempfilename=None,dataset_dir="dataset",xyzfilename="md",cutoff=5,stepinterval=1,n_clusters=10000,qmkeywords="%nproc=4\n#force mn15/6-31g(d,p)"):
        print("MDDatasetMaker")
        print("Author: Jinzhe Zeng")
        print("Email: njzjz@qq.com 10154601140@stu.ecnu.edu.cn")
        self.dumpfilename=dumpfilename
        self.bondfilename=bondfilename
        self.moleculefilename=moleculefilename if moleculefilename else self.bondfilename+".moname"
        self.tempfilename=tempfilename if tempfilename else self.bondfilename+".temp2"
        self.dataset_dir=dataset_dir
        self.xyzfilename=xyzfilename
        self.atomname=np.array(atomname)
        self.clusteratom=clusteratom
        self.atombondtype=[]
        self.trajatom_dir="trajatom"
        self.stepinterval=stepinterval
        self.ReacNetGenerator=ReacNetGenerator(atomname=self.atomname,runHMM=False,inputfilename=self.bondfilename,moleculefilename=self.moleculefilename,moleculetemp2filename=self.tempfilename,stepinterval=self.stepinterval)
        self.convertxyz=self.ReacNetGenerator.convertxyz
        self.nuclearcharge={"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10}
        self.cutoff=cutoff
        self.n_clusters=n_clusters
        self.writegjf=True
        self.gjfdir=self.dataset_dir+"_gjf"
        self.qmkeywords=qmkeywords

    def produce(self,semaphore,list,parameter):
        for item in list:
            semaphore.acquire()
            yield item,parameter

    def makedataset(self,processtraj=True,writegjf=True):
        self.writegjf=writegjf
        timearray=self.printtime([])
        for runstep in range(6):
            if runstep==0:
                if processtraj:
                    print("Run ReacNetGenerator......")
                    self.ReacNetGenerator.inputfilename=self.bondfilename
                    self.ReacNetGenerator.run()
            elif runstep==1:
                self.readlammpscrdN()
                self.readmoname()
            elif runstep==2:
                self.processatomtypelist(self.sorttrajatom)
            elif runstep==3:
                self.processatomtypelist(self.writecoulumbmatrix)
            elif runstep==4:
                self.processatomtypelist(self.selectallatom)
            elif runstep==5:
                self.mkdir(self.dataset_dir)
                if self.writegjf:
                    self.mkdir(self.gjfdir)
                self.processatomtypelist(self.writexyzfile)
            gc.collect()
            timearray=self.printtime(timearray)

    def printtime(self,timearray):
        timearray.append(time.time())
        if len(timearray)>1:
            print("Step ",len(timearray)-1," has been completed. Time consumed: ",round(timearray[-1]-timearray[-2],3),"s")
        return timearray

    def readmoname(self):
        self.mkdir(self.trajatom_dir)
        with open(self.moleculefilename) as fm,open(self.tempfilename) as ft:
            for linem,linet in zip(fm,ft):
                sm=linem.split()
                st=linet.split()
                mname=sm[0]
                matoms=np.array([int(x) for x in sm[1].split(",")])
                mbonds=np.array([[int(y) for y in x.split(",")] for x in sm[2].split(";")]) if len(sm)==3 else np.array([])
                for atom in matoms:
                    if self.atomname[self.atomtype[atom]-1] in self.clusteratom:
                        atombond=[]
                        for bond in mbonds:
                            if bond[0]==atom or bond[1]==atom:
                                atombond.append(bond[2])
                        atombondstr="".join(str(x) for x in sorted(atombond))
                        bondtype=self.atomname[self.atomtype[atom]-1]+atombondstr
                        with open(self.trajatom_dir+"/trajatom."+self.atomname[self.atomtype[atom]-1]+atombondstr,'a' if bondtype in self.atombondtype else 'w') as f:
                            print(atom,st[-1],file=f)
                        if not bondtype in self.atombondtype:
                            self.atombondtype.append(bondtype)
        with open("trajatom.list","w") as f:
            for atombondstr in  self.atombondtype:
                print(atombondstr,file=f)

    def sorttrajatom(self,trajatomfilename):
        dstep={}
        with open(self.trajatom_dir+"/trajatom."+trajatomfilename) as f:
            for line in f:
                s=line.split()
                steps=np.array([int(x) for x in s[-1].split(",")])
                for step in steps:
                    if step in dstep:
                        dstep[step].append(s[0])
                    else:
                        dstep[step]=[s[0]]
        with open(self.trajatom_dir+"/stepatom."+trajatomfilename,"w") as f:
            for step,atoms in dstep.items():
                print(step,",".join(atoms),file=f)

    def trajlist(self):
        with open("trajatom.list") as f:
            for line in f:
                gc.collect()
                yield line.strip()

    def processatomtypelist(self,func):
        for line in self.trajlist():
            func(line)

    def readlammpscrdstep(self,item):
        (step,lines),_=item
        atomtype=np.zeros((self.ReacNetGenerator.N+1),dtype=np.int)
        atomcrd=np.zeros((self.ReacNetGenerator.N+1,3))
        boxsize=[]
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
                        atomtype[int(s[0])]=int(s[1])
                        atomcrd[int(s[0])]=float(s[2]),float(s[3]),float(s[4])
                    elif linecontent==2:
                        s=line.split()
                        boxsize.append(float(s[1])-float(s[0]))
        return atomtype,atomcrd,np.array(boxsize)

    def readlammpscrdN(self):
        self.ReacNetGenerator.inputfilename=self.bondfilename
        self.bondsteplinenum=self.ReacNetGenerator.readlammpsbondN()
        self.ReacNetGenerator.inputfilename=self.dumpfilename
        self.steplinenum=self.ReacNetGenerator.readlammpscrdN()
        self.N=self.ReacNetGenerator.N
        self.atomtype=self.ReacNetGenerator.atomtype

    def writecoulumbmatrix(self,trajatomfilename):
        self.dstep={}
        with open(self.trajatom_dir+"/stepatom."+trajatomfilename) as f:
            for line in f:
                s=line.split()
                self.dstep[int(s[0])]=[int(x) for x in s[1].split(",")]
        with open(self.dumpfilename) as f,open(self.trajatom_dir+"/coulumbmatrix."+trajatomfilename,'w') as fm,Pool(maxtasksperchild=100) as pool:
            semaphore = Semaphore(360)
            results=pool.imap_unordered(self.writestepmatrix,self.produce(semaphore,enumerate(itertools.islice(itertools.zip_longest(*[f]*self.steplinenum),0,None,self.stepinterval)),None),10)
            for result in results:
                for resultline in result:
                    print(resultline,file=fm)
                semaphore.release()

    def writestepmatrix(self,item):
        (step,lines),_=item
        results=[]
        if step in self.dstep:
            atomtype,atomcrd,boxsize=self.readlammpscrdstep(item)
            for atoma in self.dstep[step]:
                cutoffatoms=[]
                for i in range(len(atomcrd)):
                    dxyz=atomcrd[atoma]-atomcrd[i]
                    dxyz=dxyz-np.round(dxyz/boxsize)*boxsize
                    if np.linalg.norm(dxyz)<=self.cutoff:
                        cutoffatoms.append(i)
                cutoffcrds=atomcrd[cutoffatoms]
                for j in range(len(cutoffcrds)):
                    cutoffcrds[j]-=np.round((cutoffcrds[j]-atomcrd[atoma])/boxsize)*boxsize
                results.append(" ".join([str(step),str(atoma),",".join(str(x) for x in self.calcoulumbmatrix(atomtype[cutoffatoms],cutoffcrds))]))
        return results

    def calcoulumbmatrix(self,atomtype,atomcrd):
        return -np.sort(-np.linalg.eig([[self.nuclearcharge[self.atomname[atomtype[i]-1]]**2.4/2 if i==j else self.nuclearcharge[self.atomname[atomtype[i]-1]]*self.nuclearcharge[self.atomname[atomtype[j]-1]]/np.linalg.norm(atomcrd[i]-atomcrd[j]) for j in range(len(atomcrd))] for i in range(len(atomcrd))])[0])

    def selectatoms(self,trajatomfilename):
        coulumbmatrix=[]
        stepatom=[]
        maxsize=0
        with open(self.trajatom_dir+"/coulumbmatrix."+trajatomfilename) as f:
            for line in f:
                s=line.split()
                stepatom.append([s[x] for x in range(2)])
                mline=[float(x) for x in s[2].split(",")]
                maxsize=max(maxsize,len(mline))
                coulumbmatrix.append(mline)
        chooseindexs=self.clusterdatas(np.array([mline+[0]*(maxsize-len(mline)) for mline in coulumbmatrix]),n_clusters=self.n_clusters) if len(coulumbmatrix)>self.n_clusters else range(len(coulumbmatrix))
        with open(self.trajatom_dir+"/chooseatoms."+trajatomfilename,'w') as f:
            for index in chooseindexs:
                print(" ".join(stepatom[index]),file=f)

    def clusterdatas(self,X,n_clusters=10000):
        clus=MiniBatchKMeans(n_clusters=n_clusters)
        labels=clus.fit_predict(X)
        chooseindex={}
        choosenum={}
        for index,label in enumerate(labels):
            if label in chooseindex:
                r=np.random.randint(0,choosenum[label]+1)
                if r==0:
                    chooseindex[label]=index
                choosenum[label]+=1
            else:
                chooseindex[label]=index
                choosenum[label]=0
        return chooseindex.values()

    def mkdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def writexyzfile(self,trajatomfilename):
        self.dstep={}
        with open(self.trajatom_dir+"/chooseatoms."+trajatomfilename) as f:
            for line in f:
                s=line.split()
                if int(s[0]) in self.dstep:
                    self.dstep[int(s[0])].append(int(s[1]))
                else:
                    self.dstep[int(s[0])]=[int(s[1])]
        with open(self.dumpfilename) as f,open(self.bondfilename) as fb,Pool(maxtasksperchild=100) as pool:
            semaphore = Semaphore(360)
            i=0
            results=pool.imap_unordered(self.writestepxyzfile,self.produce(semaphore,enumerate(zip(itertools.islice(itertools.zip_longest(*[f]*self.steplinenum),0,None,self.stepinterval),itertools.islice(itertools.zip_longest(*[fb]*self.bondsteplinenum),0,None,self.stepinterval))),None),10)
            for result in results:
                for resultline in result:
                    self.convertxyz(resultline[1],resultline[0],self.dataset_dir+"/"+self.xyzfilename+"_"+trajatomfilename+"_"+str(i)+".xyz")
                    if self.writegjf:
                        self.convertgjf(resultline[1],resultline[0],resultline[2],self.gjfdir+"/"+self.xyzfilename+"_"+trajatomfilename+"_"+str(i)+".gjf")
                    i+=1
                semaphore.release()

    def convertgjf(self,types,crds,oxygennum,gjffilename):
        with open(gjffilename,'w') as f:
            print(self.qmkeywords,file=f)
            print("",file=f)
            print(gjffilename,"oxygennum=",oxygennum,"by MDDatasetMaker",file=f)
            print("",file=f)
            #only support CHO
            S=[self.atomname[x-1] for x in types].count("H")%2+1+oxygennum*2
            print("0",S,file=f)
            for atomcrd,atomtype in zip(crds,types):
                print(self.atomname[atomtype-1]," ".join(str(x) for x in atomcrd),file=f)
            print("",file=f)

    def writestepxyzfile(self,item):
        (step,(dumplines,bondlines)),_=item
        results=[]
        if step in self.dstep:
            atomtype,atomcrd,boxsize=self.readlammpscrdstep(((step,dumplines),None))
            molecules=self.ReacNetGenerator.readlammpsbondstep(((step,bondlines),None))[0].keys()
            for atoma in self.dstep[step]:
                cutoffatoms=[]
                for i in range(len(atomcrd)):
                    dxyz=atomcrd[atoma]-atomcrd[i]
                    dxyz=dxyz-np.round(dxyz/boxsize)*boxsize
                    if np.linalg.norm(dxyz)<=self.cutoff:
                        cutoffatoms.append(i)
                #make cutoff atoms in molecules
                oxygennum=0
                for mo in molecules:
                    for moatom in mo[0]:
                        if moatom in cutoffatoms:
                            cutoffatoms=list(set(cutoffatoms)|set(mo[0]))
                            if len(mo[0])==2:
                                if np.array([self.atomname[atomtype[mo[0][x]]-1]=="O" for x in range(len(mo[0]))]).all()==True:
                                    oxygennum+=int(len(mo[0])/2)
                            break
                cutoffcrds=atomcrd[cutoffatoms]
                for j in range(len(cutoffcrds)):
                    cutoffcrds[j]-=np.round((cutoffcrds[j]-atomcrd[atoma])/boxsize)*boxsize
                results.append([cutoffcrds,atomtype[cutoffatoms],oxygennum])
        return results

if __name__ == '__main__':
    DatasetMaker(bondfilename="bonds.reaxc.ch4_new",dataset_dir="dataset_ch4",xyzfilename="ch4",stepinterval=25).makedataset(processtraj=False)
