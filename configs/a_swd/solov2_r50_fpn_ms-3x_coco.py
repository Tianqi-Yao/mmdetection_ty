_base_ = './solov2_r50_fpn_1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 800), (1333, 768), (1333, 736), (1333, 704),
                (1333, 672), (1333, 640)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# training schedule for 3x
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]


# ========================= Dataset Configs ==========================
data_root = 'data/swd/'  # 修改为你的数据集目录
metainfo = {
    'classes': ('swd',),
    'palette': [
        (220, 20, 60),
    ]
}

dataset_type = 'CocoDataset'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # filter_cfg=dict(filter_empty_gt=False),
        ann_file='train/train.json',  # 修改路径
        data_prefix=dict(img='train/'),
    ))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # filter_cfg=dict(filter_empty_gt=False),
        ann_file='val/val.json',  # 修改路径
        data_prefix=dict(img='val/'),
    ))

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        # filter_cfg=dict(filter_empty_gt=False),
        ann_file='test/test.json',  # 修改路径
        data_prefix=dict(img='test/'),
    ))  # 测试集不需要过滤空GT

# 评估指标
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val.json',
    metric=['bbox', 'segm'])

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric=['bbox', 'segm'])

# =============可视化分析=====================
vis_backends = [
    dict(type='LocalVisBackend'),  # 本地可视化后端
    dict(type='TensorboardVisBackend')  # TensorBoard 后端
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',

)

default_hooks = dict(
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,  # 是否绘制图像
        show=False  # 是否显示窗口（一般用于调试，训练时关闭）
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,  # 每 5 个 epoch 保存一次
        max_keep_ckpts=3,  # 最多保留 3 个检查点
        save_best='auto',  # 保存最佳模型
    ),
    # early_stopping=dict(
    #     type="EarlyStoppingHook",
    #     monitor="coco/segm_mAP",
    #     patience=10,
    #     min_delta=0.005),
)

# ====================epoch次数==========================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)

# =======================设置优化器============================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'
