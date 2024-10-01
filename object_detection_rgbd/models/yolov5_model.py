from .base_model import BaseModel
import torch
import numpy as np
from PIL import Image
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
import cv2
import logging
import warnings

warnings.filterwarnings("ignore")


class YoloV5Model(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the YoloV5Model with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing model parameters.
        """
        self.model_name = config.get("model_name", "yolov5s")
        self.device = (
            "cuda"
            if config.get("use_gpu", False) and torch.cuda.is_available()
            else "cpu"
        )
        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.iou_threshold = config.get("iou_threshold", 0.45)
        self.filter_classes = config.get("filter_classes", None)
        self.class_names = config.get("class_names", None)
        # YOLO5 uses 80 COCO classes, remove N/A and __background__
        self.class_names = [
            x for x in self.class_names if x != "N/A" and x != "__background__"
        ]
        self.model = None

        self.logger = logging.getLogger("YoloV5Model")

        if self.class_names:
            self.logger.info(
                "Custom class_names provided. They will be used for mapping."
            )
        else:
            self.logger.info(
                "No custom class_names provided. Model's built-in class names will be used."
            )

    def load(self):
        """
        Loads the YOLOv5 model onto the specified device using torch.hub.
        """
        try:
            # Available models: 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
            self.model = torch.hub.load(
                "ultralytics/yolov5", self.model_name, pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()

            if self.class_names:
                model_class_count = len(self.model.names)
                config_class_count = len(self.class_names)
                self.logger.info(f"Model class count: {model_class_count}")
                self.logger.info(f"Config class count: {config_class_count}")
                if model_class_count != config_class_count:
                    self.logger.error(
                        f"Mismatch in class counts: model has {model_class_count}, "
                        f"but config has {config_class_count}."
                    )
                    raise ValueError(
                        "class_names count does not match model's class count."
                    )
                else:
                    self.logger.info(
                        f"class_names count matches model's class count: {model_class_count}"
                    )

            self.logger.info(
                f"YOLOv5 model '{self.model_name}' loaded successfully on {self.device}."
            )

        except Exception as e:
            self.logger.error(f"Failed to load YOLOv5 model '{self.model_name}': {e}")
            raise e

    def predict(self, image):
        """
        Performs object detection on the input image.

        Args:
            image (np.ndarray): Input image in BGR format (as from OpenCV).

        Returns:
            torch.hub.Output: Model outputs containing detections.
        """
        try:
            if isinstance(image, np.ndarray):
                # Convert BGR (OpenCV) to RGB (PIL)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            elif isinstance(image, Image.Image):
                image_pil = image
            else:
                raise ValueError(
                    "Unsupported image type. Expected np.ndarray or PIL.Image."
                )

            # Inference
            results = self.model(image_pil, size=640)

            self.logger.debug(f"Number of detections: {len(results.xyxy[0])}")

            return results
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise e

    def process_outputs(self, outputs, image_size, header):
        """
        Processes the YOLOv5 outputs and converts them into Detection2DArray.

        Args:
            outputs (torch.hub.Output): Model outputs from YOLOv5.
            image_size (tuple): Tuple containing (width, height) of the image.
            header: ROS message header.

        Returns:
            Detection2DArray: Array of detections.
        """
        detections = Detection2DArray()
        detections.header = header

        try:
            if outputs is None or len(outputs.xyxy[0]) == 0:
                self.logger.debug("No detections to process.")
                return detections  # No detections

            # Extract detections
            pred = outputs.xyxy[0]  # (x1, y1, x2, y2, confidence, class)

            img_w, img_h = image_size

            for *box, score, cls in pred:
                score = score.item()
                cls = int(cls.item())

                if score < self.confidence_threshold:
                    continue

                # Retrieve class name
                if self.class_names:
                    class_name = self.class_names[cls]
                else:
                    class_name = self.model.names[cls]

                if self.filter_classes and class_name not in self.filter_classes:
                    continue

                xmin, ymin, xmax, ymax = box
                xmin = float(max(0, xmin.item()))
                ymin = float(max(0, ymin.item()))
                xmax = float(min(img_w - 1, xmax.item()))
                ymax = float(min(img_h - 1, ymax.item()))

                # Create Detection2D message
                detection = Detection2D()
                detection.header = header

                # Object hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis = ObjectHypothesis()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = float(score)
                detection.results.append(hypothesis)

                # Bounding box
                bbox_msg = BoundingBox2D()
                bbox_msg.center = Pose2D()
                bbox_msg.center.position = Point2D()
                bbox_msg.center.position.x = float((xmin + xmax) / 2.0)
                bbox_msg.center.position.y = float((ymin + ymax) / 2.0)
                bbox_msg.center.theta = 0.0

                bbox_msg.size_x = float(xmax - xmin)
                bbox_msg.size_y = float(ymax - ymin)
                detection.bbox = bbox_msg

                detections.detections.append(detection)

            self.logger.debug(f"Processed {len(detections.detections)} detections.")

            return detections

        except Exception as e:
            self.logger.error(f"Processing outputs failed: {e}")
            return detections
