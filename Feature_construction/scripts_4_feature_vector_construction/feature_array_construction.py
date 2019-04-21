import math
import os
import pdb

SRC_DIR="./NEW_tmp/"
DES_DIR="./feature_arrays/"

matrix_line1=[]
line1=[]
line1_tmp=[]
line2=[]
matrix_line2=[]
line2_tmp=[]
tmp=[]
n=0
#pdb.set_trace()
ff=open("combined_features_counter.txt", "w")
for root, dirs, files in os.walk(SRC_DIR):
    for file in files:
        print file
        f=open(SRC_DIR+file, 'r')
        # line1 has 10 elems
        for line in f:
            n=n+1
            line=line.strip()
            #print line
            if(n>1):
                line2=line1
                tmp=[float(field) for field in line.split(' ')]
                line2.extend(tmp)
                matrix_line2.append(line2)
            line1=[float(field) for field in line.split(' ')]
            matrix_line1.append(line1)
        line1_tmp=[sum(x) for x in zip(*matrix_line1)]
        line1=[x/n for x in line1_tmp]
#        print "line1:", line1
        if(n>=2):
            line2_tmp=[sum(x) for x in zip(*matrix_line2)]
            line2=[x/(n-1) for x in line2_tmp]
        else:
            line2=line1
            line2.extend(line1)
        n=0
        line1.extend(line2)
        ff.write(",".join([str(i) for i in line1]))
        ff.write("\n")
#        print "line1-2:", line1
        matrix_line1=[]
        line1=[]
        line1_tmp=[]
        line2=[]
        matrix_line2=[]
        line2_tmp=[]
        line3=[]

ff.close()
