# model_classes/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate_response(self, user_message: str):
        pass