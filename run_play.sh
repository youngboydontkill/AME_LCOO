# !/bin/bash
# python scripts/rsl_rl/play.py \
# --task AME-G1-29DOF-Play-v0 \
# --checkpoint logs/rsl_rl/g1_ame/2026-08-03_21-47-33/model_10000.pt \
# --num_envs 1 \
# --video \
# --video_length 300 \
# --save_attention_weights \
# --vis_attention \


python scripts/rsl_rl/play.py \
--task AME-Kuavo-S46-Play-v0 \
--checkpoint logs/rsl_rl/kuavo_s46_ame/2026-08-06_15-20-53/model_13000.pt \
--num_envs 1 \
--video \
--video_length 300 \
--save_attention_weights \
--vis_attention 

# python scripts/rsl_rl/play.py \
# --task Baseline-Kuavo-S46-Play-v0 \
# --checkpoint logs/rsl_rl/kuavo_s46_baseline/2026-08-08_19-28-07/model_9999.pt \
# --num_envs 1 \
# --video \
# --video_length 300 \