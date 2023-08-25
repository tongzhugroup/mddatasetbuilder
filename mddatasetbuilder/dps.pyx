# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# cython: infer_types=True
"""Connect molecule with Depth-First Search."""
from libc.stdlib cimport malloc, free

import cython

cdef extern from "c_stack.h":
    # This function is copied from https://zhuanlan.zhihu.com/p/38212302
    cdef cppclass C_Stack:
        void push(int val)
        int pop()


@cython.binding(False)
def dps(bonds):
    molecule = []
    cdef int _N = len(bonds)
    cdef int *visited = <int *> malloc(_N * sizeof(int))
    cdef int i, s, b_c, nbonds, ib
    cdef C_Stack st
    for i in range(_N):
        visited[i]=0

    for i in range(_N):
        if visited[i]==0:
            mol = []
            st.push(i)
            while True:
                s = st.pop()
                if s < 0:
                    break
                elif visited[s]==1:
                    continue
                mol.append(s)
                nbonds = len(bonds[s])
                for ib in range(nbonds):
                    b_c = bonds[s][ib]
                    if visited[b_c]==0:
                        st.push(b_c)
                visited[s]=1
            molecule.append(mol)
    free(visited)
    return molecule
