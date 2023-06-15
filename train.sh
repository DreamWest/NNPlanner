TRAIN_SCRIPT=train_simple_policy_avg_pool.py
PRETRAINED_EPOCHS=300
EPOCHS=300
BATCH_SIZE=16
N_STEPS=5
WORK_DIR=trial_0

# Exp1:
# EXP_NAME=test_rel_goal_norm_by_env
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)"

# Exp2:
# EXP_NAME=test_rel_goal_norm_by_env_with_attitude
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-attitude

# Exp3:
# EXP_NAME=test_rel_goal_norm_by_env_with_attitude_and_angvel
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-attitude --use-angvel

# Exp4:
# EXP_NAME=test_rel_goal_norm_by_horizon
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --norm-goal-by-horizon

# Exp5:
# EXP_NAME=test_rel_goal_norm_by_horizon_with_attitude
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --norm-goal-by-horizon --use-attitude

# Exp6:
# EXP_NAME=test_rel_goal_norm_by_horizon_with_attitude_and_angvel
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --norm-goal-by-horizon --use-attitude --use-angvel

# Exp7:
# EXP_NAME=test_global_frame_norm_by_env
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-global-frame

# Exp8:
# EXP_NAME=test_global_frame_norm_by_env_with_attitude
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-global-frame --use-attitude

# Exp9:
# EXP_NAME=test_global_frame_norm_by_env_with_attitude_and_angvel
# python $TRAIN_SCRIPT --log --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-global-frame --use-attitude --use-angvel

# EXP_NAME=global_frame_norm_by_env_with_attitude
# EPOCHS=500
# python $TRAIN_SCRIPT --log --save --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
#     --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-global-frame --use-attitude

EXP_NAME=global_frame_norm_by_env_with_attitude_and_angvel
EPOCHS=500
python $TRAIN_SCRIPT --log --save --num-pretrained-epochs $PRETRAINED_EPOCHS --num-epochs $EPOCHS --lr 1e-4 --batch-size $BATCH_SIZE --n-steps $N_STEPS \
    --work-dir $WORK_DIR --exp-name $EXP_NAME --remarks "conv net (32-->512)" --use-global-frame --use-attitude --use-angvel