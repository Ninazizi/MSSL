_base_ = [
    '../_base_/models/simsiam.py',
    '../_base_/datasets/sem_mocov2.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# set base learning rate
# lr = 0.05
lr = 0.05/8*2*2*4
data = dict(
    samples_per_gpu=64*2*4,
    workers_per_gpu=12)
# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# optimizer
optimizer = dict(lr=lr, paramwise_options={'predictor': dict(fix_lr=True)})

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=50, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
fp16 = dict(loss_scale=512.)
