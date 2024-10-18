from .model import Model
from .model.baseline import BaselineModel
import polars as pl
from pathlib import Path

def train(model: Model, name: str):
    plan = pl.scan_parquet("data/train.parquet")
    score = model.train(plan)
    Path("models").mkdir(parents=True, exist_ok=True)
    model.save(f"models/{name}.json")
    score.write_json(f"models/score_{name}.json")


def train_baseline():
    model = BaselineModel()
    train(model, "baseline")

