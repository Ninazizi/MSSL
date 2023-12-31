# model settings
model = dict(
    type='SimSiam',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512,
        hid_channels=512,
        out_channels=512,
        num_layers=3,
        with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=512,
            hid_channels=512,
            out_channels=512,
            with_avg_pool=False,
            with_last_bn=False,
            with_last_bias=True)))
