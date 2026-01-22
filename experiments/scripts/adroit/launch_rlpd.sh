python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project baselines-section \
--num_offline_steps 0 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--offline_data_ratio 0.5 \
--utd 4 \
--batch_size $((256 * 4)) \
--warmup_steps 5_000 \
$@
