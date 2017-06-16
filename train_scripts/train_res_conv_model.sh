python train.py --train_data_pattern=$data_dir --train_dir=$model_dir \
    --model='resav_ConvModel' \
    --start_new_model \
    --feature_names='rgb,audio' --feature_sizes='1024,128' \
    --moe_num_mixtures=8 --frame_features=True \
    --batch_size=256 \
    --base_learning_rate=0.001 --num_epochs=25 \
    --moe_num_hiddens=512 --conv_hidden1=1024 --conv_hidden2=512 --conv_hidden3=512 \
    --video_level_classifier_model='maxout_MoeModel'
