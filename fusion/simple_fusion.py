#!/usr/bin/python2

# if not specified, learning rate for RNN = 0.001, for others = 0.1
# `full` in dir name means using both training & validation set to train the model

# git branch `diff`: using the difference between adjacent frame pair as model input
# git branch `filter`: only using part of the categories

# frame_level DBOF model | iter 16958
f2="/DATACENTER/3/xiw/yt8m/frame_level_dbof_rgb_audio/predictions-16958.csv"
# frame_level LSTM model | iter 72020
f3="/DATACENTER/3/xiw/yt8m/frame_level_lstm_rgb_audio/predictions-72020.csv"
# video_level MOE model | iter 23010
f4="/DATACENTER/3/xiw/yt8m/video_level_moe_rgb_audio/predictions.csv"
# frame_level GRU model | iter 98465
f5="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions_98465.csv"
# frame_level LayerNorm LSTM model (dropout = 0.75)| iter 158413
f6="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio/predictions.csv"
# frame_level GRU model (using the difference between adjacent frame pair as model input) | iter 107961
f7="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable/predictions.csv"
# frame_level GRU model (using the difference between adjacent frame pair as model input) (learning rate = 0.0005) | iter 124777
f8="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_lrd2/predictions.csv"
# frame_level GRU model (using the difference between adjacent frame pair as model input) | iter 107961
f9="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50)| iter 146268
f10="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50/predictions_146268.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50) | iter 144006
f11="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/prediction_144006.csv"
# video_level MOE model (only using 3571 categories) | iter 27400
f12="/DATACENTER/3/xiw/yt8m/video_level_moe_rgb_audio_full_filter/prediction.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50) | iter 203360
f13="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-203360.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50) | iter 240360
f14="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-240360.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50) | iter 222360
f15="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-222360.csv"
# frame_level GRU model | iter 60860
f16="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions-60860.csv"
# frame_level GRU model | iter 80297
f17="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions-80297.csv"
# frame_level LayerNorm LSTM model (dropout = 0.75)| iter 150168
f18="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio/predictions-lnblstm-150168.csv"
# frame_level GRU model (using the difference between adjacent frame pair as model input) | iter 80177
f19="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions-80177.csv"
# frame_level GRU model (using the difference between adjacent frame pair as model input) | iter 60954
f20="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions-60954.csv"
# frame_level LayerNorm LSTM model (dropout = 0.50) | iter 286374
f21="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-286k.csv"
# video_level MOE model (only using 2534 categories) | iter 27177
f22="/DATACENTER/3/xiw/yt8m/video_level_moe_rgb_audio_full_filter-2534_r05/predictions.csv"

f = [f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22]
r = [0.5, 0.1, 0.7, 1.6, 1.3, 0.3, 0.3, 0.6, 1.1, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.1, 0.1, 0.8, 0.8]

import numpy as np

#r = [x / sum(rraw) for x in rraw]

print 'VideoId,LabelConfidencePairs'

res = {}
ci = 0
for fi in f:
  with open(fi) as file:
    ca = file.readlines()
  cc = [x.strip().split(',') for x in ca[1:]]

  for k in range(len(cc)):
    if cc[k][0] in res:
      d = res[cc[k][0]]
    else:
      d = {}
    t=100
    id = cc[k][1].split()[:t:2]
    val = cc[k][1].split()[1:t:2]
    for i in range(len(id)):
      pred = float(val[i])
      if id[i] in d:
        d[id[i]] += pred * r[ci]
      else:
        d[id[i]] = pred * r[ci]
    res[cc[k][0]] = d
  ci += 1

for n in res:
  p = n +','
  for id, value in sorted(res[n].iteritems(), key=lambda (k,v): (-v,k))[:30]:
    p += id + ' ' + "%f" % value + ' '
  print p
