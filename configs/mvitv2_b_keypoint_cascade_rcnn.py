from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2 import model_zoo
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT
from .coco_loader_nolsj import dataloader
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import KRCNNConvDeconvUpsampleHead

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(
    embed_dim=96,
    depth=10,
    num_heads=1,
    last_block_indexes=(0, 2, 7, 9),
    residual_pooling=True,
    drop_path_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=("scale2", "scale3", "scale4", "scale5"),
)
model.backbone.in_features = "${.bottom_up.out_features}"


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth"
n = 2
dataloader.train.total_batch_size = 64 //n

# 36 epochs
train.max_iter = 67500 * n
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[52500*n, 62500*n, 67500*n],
    ),
    warmup_length=250*n / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 1.6e-4 / n


# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm=lambda c: NaiveSyncBatchNorm(c, stats_mode="N"),
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# Using NaiveSyncBatchNorm becase heads may have empty input. That is not supported by
# torch.nn.SyncBatchNorm. We can remove this after
# https://github.com/pytorch/pytorch/issues/36530 is fixed.
model.roi_heads.mask_head.conv_norm = lambda c: NaiveSyncBatchNorm(c, stats_mode="N")

# 2conv in RPN:
# https://github.com/tensorflow/tpu/blob/b24729de804fdb751b06467d3dce0637fa652060/models/official/detection/modeling/architecture/heads.py#L95-L97  # noqa: E501, B950
model.proposal_generator.head.conv_dims = [-1, -1]


model.backbone.bottom_up.depth = 24
model.backbone.bottom_up.last_block_indexes = (1, 4, 20, 23)
model.backbone.bottom_up.drop_path_rate = 0.4

train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_mvitv2_b_in21k/f325820315/model_final_8c3da3.pkl"
train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_b_in12k_3x/f309003202/model_final_be5168.pkl"
model.roi_heads.update(
    num_classes=1,
    keypoint_in_features=["p2", "p3", "p4", "p5"],
    keypoint_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    keypoint_head=L(KRCNNConvDeconvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_keypoints=17,
        conv_dims=[512] * 8,
        loss_normalizer="visible",
    ),
)