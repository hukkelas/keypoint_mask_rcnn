from pathlib import Path

import click
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.transforms import AugmentationList
from detectron2.engine import DefaultPredictor
from detectron2.modeling.roi_heads import CascadeROIHeads, StandardROIHeads
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from PIL import Image


class KeypointDetector:

    def __init__(self, config_file: str, model_url: str, score_threshold: float, device) -> None:
        assert Path(config_file).is_file(), f"{config_file} does not exist."
        
        cfg = LazyConfig.load(config_file)
        if cfg.model.roi_heads._target_ == CascadeROIHeads:
            for head in cfg.model.roi_heads.box_predictors:
                head.test_score_thresh = score_threshold
        else:
            assert cfg.model.roi_heads._target_ == StandardROIHeads
            cfg.model.roi_heads.box_predictor.test_score_thresh = score_threshold
            
        self.model = instantiate(cfg.model).eval().to(device)
        DetectionCheckpointer(self.model).load(model_url)
        self.aug = instantiate(cfg.dataloader.test.mapper.augmentations)
        assert len(self.aug) == 1
        self.aug = self.aug[0]
        self.image_format = cfg.dataloader.train.mapper.image_format

    def predict(self, im: np.ndarray) -> Instances:
        if self.image_format == "BGR":
            im = im[:, :, ::-1]
        H, W = im.shape[:2]
        im = self.aug.get_transform(im).apply_image(im)
        im = torch.as_tensor(im.astype(np.float32).transpose(2, 0, 1))
        inputs = dict(image=im, height=H, width=W)
        instances = self.model([inputs])[0]["instances"]
        # instances contains 
        # dict_keys(['pred_boxes', 'scores', 'pred_classes', 'pred_masks', 'pred_keypoints', 'pred_keypoint_heatmaps'])
        return instances

    def visualize_prediction(self, im, instances):
        visualizer = Visualizer(im)
        instances = instances.to("cpu")
        return visualizer.draw_instance_predictions(predictions=instances).get_image()



@torch.no_grad()
@click.command()
@click.argument("impath")
@click.option("--config-file", default="configs/keypoint_maskrcnn_R_50_FPN_1x.py")
@click.option("--model-url", default="https://folk.ntnu.no/haakohu/checkpoints/maskrcnn_keypoint/keypoint_maskrcnn_R_50_FPN_1x.pth")
@click.option("--score-threshold", default=.5, type=float)
def main(impath: str, config_file: str, model_url:str, score_threshold):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    detector = KeypointDetector(config_file, model_url, score_threshold, device=device)
    im = np.array(Image.open(impath).convert("RGB"))
    instances = detector.predict(im)
    visualized_prediction = detector.visualize_prediction(im, instances)
    Image.fromarray(visualized_prediction).save("images/example_output.png")


if __name__ == "__main__":
    main()
