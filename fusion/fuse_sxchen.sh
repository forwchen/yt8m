# This script fuse ensembles from team Shaoxiang Chen with the corresponding weights.
# The weights here are empirically decided.
# For the details of the models, please refer to our doc/code/paper.
python weighted_fuse.py 1,1,1,1,0.5,0.5,1,0.25,0.25 \
fuse_lstm_353k_323k_300k_280k_4321.csv \
fuse_gru_69k_65k_60k_55k_4321.csv \
fuse_rwa_114k_87k_75k_50k_4321.csv \
fuse_dgru_56k_50k_46k_40k_35k_4320505.csv \
fuse_moe2_127k_115k_102k_90k_4321.csv \
fuse_dbof_175k_150k_137k_122k_112k_4320505.csv \
fuse_latevlad_24k_21k_19k_16k_13k_4320505.csv \
fuse_bngru_86k_74k_65k_49k_4321.csv \
fuse_bigru_53k_45k_35k_532.csv
