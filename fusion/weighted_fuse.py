import sys

"""
Usage example: python weighted_fuse.py 1.0,1.0,1.0 prediction_1.csv prediction_2.csv prediction_3.csv

We usually make predictions for top 32 classes.

For each sample, this script takes predictions from several models,
compute a weighted sum of the confidence scores for all the predicted classes.
Then rearrange the prediction according to the new fused scores.

The resulted predictions may contain more than 32 classes,
because different model may predict different classes and that's the complementarity between them.
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
    # parse lines of predictions
    for l in lines:
        l = l.replace('\n', '')
        k = l.split(',')[0]
        vv = l.split(',')[1].split(' ')
        if len(vv[-1]) == 0:
            vv.pop()
        maxlen_vv = max(maxlen_vv, len(vv))

        if not preds.has_key(k):
            preds[k] = {}
            #cnts[k] = {}

        for i in range(0, len(vv), 2):
            if i % 2 == 0:
                if not preds[k].has_key(vv[i]):
                    preds[k][vv[i]] = 0
                    #cnts[k][vv[i]] = 0

                # weighted sum of scores
                preds[k][vv[i]] += float(vv[i+1]) * weights[n]
                #cnts[k][vv[i]] += 1

#for k in preds.keys():
#    for kk in preds[k].keys():
#        preds[k][kk] /= cnts[k][kk]


# generate fused predictions
fo = open('fuse.csv', 'w')
fo.write('VideoId,LabelConfidencePairs\n')
for k in preds.keys():
    d = preds[k]
    # sort with the new scores as key
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






