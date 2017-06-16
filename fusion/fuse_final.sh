# 1. fuse_temp_sxc.csv is the model ensemble from team Shaoxiang Chen
# 2. fusionv9_xiw.csv is the model ensemble from team Xi Wang
# 3. yyt_ensemble_topk30_0531.csv is the model ensemble from team Yongyi Tang
# 4. fuse_resrnn_75k_70k_62k_56k_4321.csv is another model Shaoxiang trained.
#    It uses residual connections in RNN.
# This script fuses 4 ensembles to generate our final ensemble with the corresponding weights.

python weighted_fuse_count.py 1,0.2,1,0.5 \
fuse_temp_sxc.csv \
fusionv9_xiw.csv \
yyt_ensemble_topk30_0531.csv \
fuse_resrnn_75k_70k_62k_56k_4321.csv
