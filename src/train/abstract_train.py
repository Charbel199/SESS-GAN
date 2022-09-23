from abc import ABC, abstractmethod
from conf import ModelConfig


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(self, config: ModelConfig): raise NotImplementedError
