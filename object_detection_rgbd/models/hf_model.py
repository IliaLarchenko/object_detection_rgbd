from .base_model import BaseModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    DetrImageProcessor,
    DetrForObjectDetection,
    YolosImageProcessor,
    YolosForObjectDetection,
)
import torch
import numpy as np
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    ObjectHypothesis,
    BoundingBox2D,
    Pose2D,
    Point2D,
)
from typing import Dict, Any


class HFModel(BaseModel):
    # Base model for hugging face models
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("model_name", "facebook/detr-resnet-50")
        self.device = config.get("device", "cpu")
        self.confidence_threshold = config.get("confidence_threshold", 0.9)
        self.filter_classes = config.get("filter_classes", None)
        self.class_names = config.get("class_names", None)
        self.model = None
        self.feature_extractor = None

    def load(self):
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name).to(
            self.device
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)

    def predict(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def process_outputs(self, outputs, image_size, header):
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        probas = logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.confidence_threshold

        if not keep.any():
            detections = Detection2DArray()
            detections.header = header
            return detections

        bboxes_scaled = self.rescale_bboxes(pred_boxes[0, keep], image_size)
        labels = probas[keep].argmax(-1)
        scores = probas[keep].max(-1).values

        filtered_indices = []
        for idx, label in enumerate(labels):
            class_name = self.class_names[label]
            if self.filter_classes is None or class_name in self.filter_classes:
                filtered_indices.append(idx)

        if not filtered_indices:
            detections = Detection2DArray()
            detections.header = header
            return detections

        filtered_labels = labels[filtered_indices]
        filtered_scores = scores[filtered_indices]
        filtered_bboxes = bboxes_scaled[filtered_indices]

        detections = Detection2DArray()
        detections.header = header

        for label, score, bbox in zip(
            filtered_labels, filtered_scores, filtered_bboxes
        ):
            class_name = self.class_names[label]

            detection = Detection2D()
            detection.header = header

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis = ObjectHypothesis()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = float(score.item())
            detection.results.append(hypothesis)

            bbox_msg = BoundingBox2D()
            xmin, ymin, xmax, ymax = bbox

            bbox_msg.center = Pose2D()
            bbox_msg.center.position = Point2D()
            bbox_msg.center.position.x = float((xmin + xmax) / 2.0)
            bbox_msg.center.position.y = float((ymin + ymax) / 2.0)
            bbox_msg.center.theta = 0.0

            bbox_msg.size_x = float(xmax - xmin)
            bbox_msg.size_y = float(ymax - ymin)
            detection.bbox = bbox_msg

            detections.detections.append(detection)

        return detections

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = out_bbox * torch.tensor(
            [img_w, img_h, img_w, img_h], dtype=torch.float32
        ).to(self.device)
        xmin = b[:, 0] - b[:, 2] / 2
        ymin = b[:, 1] - b[:, 3] / 2
        xmax = b[:, 0] + b[:, 2] / 2
        ymax = b[:, 1] + b[:, 3] / 2
        bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        bboxes = bboxes.detach().cpu().numpy().astype(int)

        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, img_w - 1)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, img_h - 1)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, img_w - 1)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, img_h - 1)

        return bboxes


class DetrModel(HFModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load(self):
        self.model = DetrForObjectDetection.from_pretrained(self.model_name).to(
            self.device
        )
        self.feature_extractor = DetrImageProcessor.from_pretrained(self.model_name)


class YolosModel(HFModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load(self):
        self.model = YolosForObjectDetection.from_pretrained(self.model_name).to(
            self.device
        )
        self.feature_extractor = YolosImageProcessor.from_pretrained(self.model_name)
