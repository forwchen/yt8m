MODEL_DIR=/DATACENTER/850/yt8m

CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --train_data_pattern='whatever*.tfrecord' \
    --train_list='train_val2.lst' \
    --use_grad_agg=False \
    --model=BNGRUModel \
    --train_dir=$MODEL_DIR/frame_level_bngru_model \
    --frame_features=True --feature_names="rgb,audio" --feature_sizes="1024,128" \
    --batch_size=128 \
    --num_epochs=500 \
    --base_learning_rate=0.001 \
    --learning_rate_decay_examples=6000000 \
    --clip_gradient_norm=1.0 \
    --regularization_penalty=1.0 \

