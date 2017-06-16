#!/usr/bin/python2

f1="/DATACENTER/3/xiw/yt8m/video_level_logistic_model_rgb_audio/predictions.csv"
f2="/DATACENTER/3/xiw/yt8m/frame_level_dbof_rgb_audio/predictions-16958.csv"
f3="/DATACENTER/3/xiw/yt8m/frame_level_lstm_rgb_audio/predictions-72020.csv"
f4="/DATACENTER/3/xiw/yt8m/video_level_moe_rgb_audio/predictions.csv"
f5="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions_98465.csv"
f6="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio/predictions.csv"
f7="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable/predictions.csv"
f8="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_lrd2/predictions.csv"
f9="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions.csv"
f10="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50/predictions_146268.csv"
f11="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/prediction_145000.csv"
f12="/DATACENTER/3/xiw/yt8m/video_level_moe_rgb_audio_full_filter/prediction.csv"
f13="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-200k.csv"
f14="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-240k.csv"
f15="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-225k.csv"
f16="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions-60860.csv"
f17="/DATACENTER/3/xiw/yt8m/frame_level_gru_rgb_audio/predictions-80297.csv"
f18="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio/predictions-lnblstm-150168.csv"
f19="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions-80177.csv"
f20="/DATACENTER/3/xiw/yt8m/frame_level_grud_rgb_audio_stable_full/predictions-60954.csv"
f21="/DATACENTER/3/xiw/yt8m/frame_level_lnblstm_rgb_audio_d50_full/predictions-286k.csv"
f22="/DATACENTER/3/xiw/yt8m//video_level_moe_rgb_audio_full_filter-2534_r05/predictions.csv"
f23="/DATACENTER/3/xiw/yt8m//video_level_moe_rgb_audio_full_filter-2534_r05/predictions.csv"


f = [f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23]
r = [0.0, 0.5, 0.1, 0.7, 1.6, 1.3, 0.3, 0.3, 0.6, 1.1, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.1, 0.1, 0.8, 0.8, 0.0]

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
