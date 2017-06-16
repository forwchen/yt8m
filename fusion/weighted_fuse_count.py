import sys

"""
This script does mostly the same thing as weighted_fuse.py.

Except that:
    For each sample, this scripts collect the number of times every class is predicted
    in all the model predictions. Then for those classes that is predicted in more than
    half of the models, we manually increase the confidence by 1/10 of the total weights.

    This is based on the intuition that if many different models predicted the same class,
    then this class is likely to be in the ground truth.

    This can be considered a trick. We used this trick for our final ensemble, which slightly improved our performance.
"""

preds = {}
cnts = {}
#weights = [1, 1, 0.33, 0.33]
weights = [float(x) for x in sys.argv[1].split(',')]

models = sys.argv[2:]
n_models = len(models)

assert n_models == len(weights), "weights should have length==n_models"

maxlen_vv = 0
for n in range(n_models):

    f_pred = models[n]
    lines = open(f_pred, 'r').readlines()[1:]
    for l in lines:
        l = l.replace('\n', '')
        k = l.split(',')[0]
        vv = l.split(',')[1].split(' ')
        if len(vv[-1]) == 0:
            vv.pop()
        maxlen_vv = max(maxlen_vv, len(vv))

        if not preds.has_key(k):
            preds[k] = {}
            cnts[k] = {}

        for i in range(0, min(64,len(vv)), 2):
            if i % 2 == 0:
                if not preds[k].has_key(vv[i]):
                    preds[k][vv[i]] = 0
                    cnts[k][vv[i]] = 0
                val = float(vv[i+1])
                preds[k][vv[i]] += val * weights[n]
                # accumulate the number of times predicted
                cnts[k][vv[i]] += 1


sw = sum(weights)
for k in preds.keys():
    for kk in preds[k].keys():
        # if predicted in more than half of the models, increase confidence
        if cnts[k][kk] >= n_models / 2.0:
            preds[k][kk] += sw / 10.

fo = open('fuse.csv', 'w')
fo.write('VideoId,LabelConfidencePairs\n')
for k in preds.keys():
    d = preds[k]
    sv = sorted(d, key=d.get, reverse=True)
    fo.write(k)
    fo.write(',')
    for i in range(len(sv)):
        #for i in range(min(32,len(sv))):
        fo.write(sv[i])
        fo.write(' ')
        fo.write(str(d[sv[i]]))
        fo.write(' ')
    fo.write('\n')






