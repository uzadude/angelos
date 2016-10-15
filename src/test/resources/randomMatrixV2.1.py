#import scipy.sparse
from scipy.sparse import rand
import random
import itertools
import numpy as np
#m=1000000
import time

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,rows=array.row,
             column =array.col )

for m in range(2000000,2100000,100000):
    dens=0.00001
    while(dens<0.0001):
        begin=time.time()
        print m
        print dens
        string=""
        a=rand(m, m, density=dens, format='coo', dtype=None, random_state=None)
        #a=rand(m, m, density=0.1, format='coo', dtype=None, random_state=None)
        print type(a)
        print a.row
        print a.col
        stringSTART="---8<------8<------8<------8<------8<---"
        stringEND="--->8------>8------>8------>8------>8---"
        string+=stringSTART+"\n"
        print "Time Elapsed for 1st: " + str(time.time()-begin)
        print "Start Creating String"
        begin=time.time()
        #cx = a.tocoo()
        #print len(a.row)
        #a.data.zfill(10)
        #a.data.tofile("test.txt",format="%s")
        #np.savetxt('data1.txt',np.r_[a.row,a.col,a.data],fmt='%s',delimiter=',',newline=' ')
        np.savetxt('/Users/oraviv/git/got/src/test/aggelos/data/myfile-'+str(m)+'-'+str(dens),np.c_[a.row,a.col,a.data],fmt='%010d %010d %f',delimiter=',',newline="\n")
        #save_sparse_csr("test",a)
        print "Time Elapsed for 2nd: " + str(time.time()-begin)
        dens=dens*10
