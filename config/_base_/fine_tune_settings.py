evaluation = dict(interval=500, metric='bbox',
                iou_thrs = [0.5],
                save_best='bbox_mAP',classwise=True)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=0.001,
    step=[3600])
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(interval=100000)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'),]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
# a magical seed works well in most cases for this repo!!!
# using different seeds might raise some issues about reproducibility
seed = 42
