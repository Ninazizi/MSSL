_base_ = 'simsiam_resnet18_8xb32-coslr-200e_sem.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=400)
