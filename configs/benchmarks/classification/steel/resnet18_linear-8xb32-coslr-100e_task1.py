_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/anneal.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]
# SwAV linear evaluation setting

model = dict(backbone=dict(frozen_stages=-1),
            head=dict(num_classes=13))
optimizer = dict(lr=0.3/8)


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=100, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
fp16 = dict(loss_scale=512.)
