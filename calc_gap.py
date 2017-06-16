#/usr/bin/python
import numpy as np
import sys
import eval_util

predf = [x.strip() for x in open(sys.argv[1])]
name = []
pred = np.zeros((len(predf)-1, 4716), dtype='float32')
k = 0
for x in predf[1:]:
  name += [x.split(',')[0],]
  p = x.split(',')[1].split()
  for i in range(0, len(p), 2):
    pred[k][int(p[i])] = float(p[i+1])
  k += 1

gtf = '/DATACENTER/3/xiw/yt8m/validate_labels.csv'
gta = {}
for x in open(gtf):
  l = x.strip().split(',')
  gta[l[0]] = l[1]


gt = np.zeros((len(predf)-1, 4716), dtype='float32')
k = 0
for n in name:
  p = gta[n].split()
  for i in p:
    gt[k][int(i)] = 1
  k += 1

print eval_util.calculate_gap(pred, gt, top_k=20)
