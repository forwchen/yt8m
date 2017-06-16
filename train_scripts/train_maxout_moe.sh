python train.py --train_data_pattern=$data_dir --train_dir=$model_dir \
    --model=maxout_MoeModel --feature_names='mean_rgb, mean_audio' \
    --feature_sizes='1024, 128' \
    --moe_num_mixtures=8 --num_epochs=5 \
    --moe_num_hiddens=1024 --lr=0.0005 --batch_size=2048
