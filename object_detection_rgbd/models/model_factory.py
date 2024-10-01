from typing import Dict, Any
from .base_model import BaseModel
from .hf_model import DetrModel, YolosModel
from .yolov5_model import YoloV5Model


class ModelFactory:
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        model_type = config.get("model_type", "").lower()

        if model_type == "detr":
            return DetrModel(config)
        elif model_type == "yolos":
            return YolosModel(config)
        elif model_type == "yolov5":
            return YoloV5Model(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
