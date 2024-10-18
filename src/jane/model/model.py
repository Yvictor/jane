import polars as pl
from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def train(self, plan: pl.LazyFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def predict(self, plan: pl.LazyFrame) -> pl.DataFrame:
        pass
        
    @abstractmethod
    def evaluate(self, plan: pl.LazyFrame) -> pl.DataFrame:
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def is_trained(self):
        pass
    
    