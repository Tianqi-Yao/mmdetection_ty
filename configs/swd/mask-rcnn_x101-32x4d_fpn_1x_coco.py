# 新配置继承了基本配置，并做了必要的修改
_base_ = './mask-rcnn_r101_fpn_1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')))

# 修改数据集相关配置
data_root = 'data/swd/'
metainfo = {
    'classes': ('swd'),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/val.json',
        data_prefix=dict(img='val/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/test.json',
        data_prefix=dict(img='test/')))

# ===============修改评价指标相关配置==================
val_evaluator = dict(ann_file=data_root + 'val/val.json',metric=['bbox', 'segm'], classwise=True)
test_evaluator = dict(ann_file=data_root + 'test/test.json',metric=['bbox', 'segm'], classwise=True)  # 测试集标注文件


# =============可视化分析=====================
vis_backends = [
    dict(type='LocalVisBackend'),  # 本地可视化后端
    dict(type='TensorboardVisBackend')  # TensorBoard 后端
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
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
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth'