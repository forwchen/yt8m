python train.py --train_data_pattern=$data_dir --train_dir=$model_dir \
    --model=audio_avgShort_twowayGRUModel --moe_num_mixtures=8 \
    --frame_features=True --batch_size=256 \
    --base_learning_rate=0.0005 --num_epochs=25 \
    --moe_num_hiddens=1024 --feature_names='rgb,audio' \
    --feature_sizes='1024,128' \
    --stride=10 \
    --video_level_classifier_model='linear_res_mix_act_MoeModel'
