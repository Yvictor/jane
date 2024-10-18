import polars as pl
from .model import Model
import xgboost as xgb


class BaselineModel(Model):
    def __init__(self):
        self._is_trained = False

    def train(self, plan: pl.LazyFrame) -> pl.DataFrame:
        n_dates = plan.select(
            pl.col("date_id").unique().count().alias("n_dates"),
        ).collect()["n_dates"][0]
        train_ratio = 0.75
        plan_train_set = plan.filter(pl.col("date_id") <= int(n_dates * train_ratio))
        plan_valid_set = plan.filter(pl.col("date_id") > int(n_dates * train_ratio))
        plan_train_feats = plan_train_set.select(pl.col("^feature_.*$"))
        plan_valid_feats = plan_valid_set.select(pl.col("^feature_.*$"))
        plan_train_target = plan_train_set.select(pl.col("responder_6"))
        plan_valid_target = plan_valid_set.select(pl.col("responder_6"))
        plan_train_weight = plan_train_set.select(pl.col("weight"))
        plan_valid_weight = plan_valid_set.select(pl.col("weight"))

        dtrain = xgb.DMatrix(
            plan_train_feats.collect(),
            label=plan_train_target.collect(),
            weight=plan_train_weight.collect(),
            feature_names=plan_train_feats.collect_schema().names(),
        )
        dvalid = xgb.DMatrix(
            plan_valid_feats.collect(),
            label=plan_valid_target.collect(),
            weight=plan_valid_weight.collect(),
            feature_names=plan_valid_feats.collect_schema().names(),
        )
        param = {
            "objective": "reg:squarederror",
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "nthread": 32,
            "verbosity": 1,
        }
        num_boost_round = 500
        model = xgb.train(
            param,
            dtrain,
            num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=20,
        )
        self._model = model
        self._is_trained = True
        df_train_score = self.evaluate(plan_train_set)
        print(df_train_score)
        df_valid_score = self.evaluate(plan_valid_set)
        print(df_valid_score)
        return df_valid_score

    def predict(self, plan: pl.LazyFrame) -> pl.Series:
        if not self._is_trained:
            raise Exception("Model is not trained")
        plan_set = plan.select(pl.col(self._model.feature_names))
        data = xgb.DMatrix(
            plan_set.collect(),
            weight=plan.select(pl.col("weight")).collect(),
            feature_names=self._model.feature_names,
        )
        pred = self._model.predict(data)
        return pl.Series("pred", pred)

    def evaluate(self, plan: pl.LazyFrame) -> pl.DataFrame:
        if not self._is_trained:
            raise Exception("Model is not trained")

        pred = self.predict(plan)
        score = (
            plan.select(
                pl.col("date_id"),
                pl.col("responder_6").alias("target"),
                pl.col("weight").alias("w"),
                pred,
            )
            .select(
                pl.col("date_id").min().alias("date_start"),
                pl.col("date_id").max().alias("date_end"),
                pl.col("date_id").count().alias("n_samples"),
                (
                    1
                    - ((pl.col("target") - pl.col("pred")) ** 2 * pl.col("w")).sum()
                    / ((pl.col("target") ** 2 * pl.col("w")).sum())
                ).alias("r2"),
            )
            .collect()
        )
        return score

    def save(self, path: str):
        if not self._is_trained:
            raise Exception("Model is not trained")
        self._model.save_model(path)

    def load(self, path: str):
        self._model = xgb.Booster()
        self._model.load_model(path)
        self._is_trained = True

    def is_trained(self):
        return self._is_trained
