_base_ = './rtmdet-ins_l_8xb32-300e_coco.py'

model = dict(
    backbone=dict(deepen_factor=0.67, widen_factor=0.75),
    neck=dict(in_channels=[192, 384, 768], out_channels=192, num_csp_blocks=2),
    bbox_head=dict(in_channels=192, feat_channels=192, num_classes=1))


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
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'
