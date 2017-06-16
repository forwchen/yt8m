import sys
import os

"""
Usage: python select_top.py prediction.csv top_k

Preserve only the top_k predictions for each sample.
This is mainly for generating a small submission file.

"""

in_file = sys.argv[1]
top_k = int(sys.argv[2])

outfile = open('top-'+str(top_k)+'-'+in_file, 'w')

lines = open(in_file, 'r').readlines()
outfile.write(lines[0])

for i in lines[1:]:
    l = i.replace('\n', '').split(',')
    key = l[0]
    val = l[1].split(' ')
    if len(val[-1]) == 0:
        val.pop()
    outfile.write(key+',')
    for j in range(min(top_k, len(val)/2)):
        outfile.write(val[j*2]+' '+val[j*2+1]+' ')
    outfile.write('\n')
outfile.close()



