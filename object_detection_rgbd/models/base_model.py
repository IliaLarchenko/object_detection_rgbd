from abc import ABC, abstractmethod

# Base model for object detection
# You can use any 2d object detection model if it outputs the correct format


class BaseModel(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass
