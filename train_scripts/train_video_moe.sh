MODEL_DIR=/DATACENTER/2/forwchen/mnt19

CUDA_VISIBLE_DEVICES=1 python train.py \
    --train_data_pattern='whatever*.tfrecord' \
    --train_list='train_val.lst' \
    --use_grad_agg=False \
    --model=MoeModel \
    --moe_num_mixtures=16 \
    --train_dir=$MODEL_DIR/video_level_moe_model \
    --frame_features=False --feature_names="mean_rgb,mean_audio" --feature_sizes="1024,128" \
    --batch_size=1024 \
    --num_epochs=500 \
    --export_model_steps=5000 \
    --base_learning_rate=0.01 \
    --learning_rate_decay_examples=8000000 \
    --clip_gradient_norm=-1.0 \
    --regularization_penalty=1.0 \

