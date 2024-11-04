_base_ = ['./redet_re50_refpn_1x_dota_le90.py']

data_root = '/home/f523/guazai/disk3/shangxiping/redet/hrsc'
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(896, 896)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + '/labeltxt/',
        img_prefix=data_root + '/images/'),
    val=dict(
        ann_file=data_root + '/labeltxt/',
        img_prefix=data_root + '/images/'),
    test=dict(
        ann_file=data_root + '/labeltxt/',
        img_prefix=data_root + '/images/')
)

model = dict(train_cfg=dict(rpn=dict(assigner=dict(gpu_assign_thr=200))))
runner = dict(type='EpochBasedRunner', max_epochs=36)