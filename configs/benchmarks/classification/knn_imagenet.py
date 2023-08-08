data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pipeline = [
    dict(type='Resize', size=224),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/mmselfsup-0.9.0/data_collect/crops/',
            ann_file=None),
        pipeline=pipeline),
   )
