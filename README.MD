# High-performance keypoint and mask detection models (detectron2)

This repository provides pre-trained **Mask-Keypoint RCNN** that predicts instance mask, keypoints and boxes.
All models are trained with [detectron2](https://github.com/facebookresearch/detectron2).


### Why?
The pre-trained keypoint R-CNN models in the detectron2 model zoo do not have mask heads and, thus, only predict keypoints and boxes (not instance masks).


## Install
Follow the [detectron2 installation instructions](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

The following works for me:
```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Pre-trained models

| Name                                                                                                                                                                                      | pre-train  | inference time (s/im) | box AP | kp. AP | mask AP | Weight path                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------- | ------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [R-50 FPN](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml) (detectron2 pre-trained)                                       | IN1k       | --                    | 53.6   | 64.0   | --*     | [[weight path]](https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl)                                                                                      |
| [R-50 FPN-3x](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml) (detectron2 pre-trained)                                    | IN1k       | --                    | 55.4   | 65.5   | --*     | [[weight path]](https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl)                                                                                      |
| [R-50 Mask R-CNN FPN-1x](configs/keypoint_maskrcnn_R_50_FPN_1x.py) (ours)                                                                                                                 | IN1k       | --                    | 55.1   | 65.3   | 47.9    | [[weight path]](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/keypoint_maskrcnn_R_50_FPN_1x.pth) [[metrics]](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/keypoint_maskrcnn_R_50_FPN_1x.json)  |
| [MViTv2-B Cascade Mask R-CNN (ours) [[original config]](https://github.com/facebookresearch/detectron2/blob/main/projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep.py)] | IN21K, sup, COCO | --                    | 65.6   | 67.2   | 55.0    | [[weight path]](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/mvitv2_b_keypoint_cascade_rcnn.pth) [[metrics]](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/mvitv2_b_keypoint_cascade_rcnn.json) |
| [regnety_4gf_FPN](configs/regnety_4gf_FPN.py) [original config](https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py) | COCO | -- | 59.4 | 67.0 | 51.4 | [weight path](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/regnety_4gf_FPN.pth) [metrics](https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/regnety_4gf_FPN.json) |

*: The pre-trained detectron2 Keypoint R-CNN has no mask head.

## Usage
You can directly use [`detect.py`](detect.py):
```
python3 detect.py path_to_image.png
```

Or, in python you can simply use the following lines:
```python
from detect import KeypointDetector
config_file = "configs/keypoint_maskrcnn_R_50_FPN_1x.py"
model_url = "https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/keypoint_maskrcnn_R_50_FPN_1x.pth"
score_threshold = .5
impath = # a path to an image.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu
detector = KeypointDetector(config_file, model_url, score_threshold, device)
im = np.array(Image.open(impath).convert("RGB"))
instances = detector.predict(im)
visualized_prediction = detector.visualize_prediction(im, instances)
Image.fromarray(visualized_prediction).show()
```


## Training
To train the network, use [`lazyconfig_train_net.py`](https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py) found in the detectron2 repository.

