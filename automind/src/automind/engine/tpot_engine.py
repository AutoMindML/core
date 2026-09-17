import json
import pickle
from pathlib import Path
from typing import Any

from pandas import DataFrame
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from tpot.tpot import TPOTClassifier, TPOTRegressor


class TPOTEngine:
    def __init__(
        self,
        model_path: str | Path = "tpot_best_pipeline.pkl",
        random_state: int = 42,
    ) -> None:
        self.model = None
        self.task_type: str | None = None
        self.input_features: list[str] = []
        self.output_features: list[str] = []
        self.model_path = Path(model_path)
        self.random_state = random_state
        self.scorer_names: list[str] = []
        self.scores: dict[str, float] = {}
        self.category_mappings: dict[str, dict[Any, int]] = {}
        self.search_config: dict[str, Any] = {}

    def train(self, df: DataFrame, target_col: str, args=None):
        args = dict(args or {})
        mode = args.pop("mode", "classification").lower()
        classification_scorers = args.pop(
            "classification_scorers", "accuracy,f1"
        ).split(",")
        regression_scorers = args.pop(
            "regression_scorers", "neg_root_mean_squared_error"
        ).split(",")
        validation_size = args.pop("validation_size", 0.25)

        self.task_type = mode
        self.scorer_names = (
            regression_scorers if mode == "regression" else classification_scorers
        )
        self.output_features = [target_col]

        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()
        self.input_features = X.columns.tolist()
        self.category_mappings = self._fit_category_mappings(X)
        X = self._encode_categories(X)

        stratify = y if mode != "regression" else None
        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        search_config = {
            "max_time_mins": args.pop("max_time_mins", 5),
            "random_state": self.random_state,
            "generations": args.pop("generations", 100),
            "population_size": args.pop("population_size", 100),
            "cv": args.pop("cv", 5),
            "n_jobs": args.pop("n_jobs", 1),
            "verbosity": args.pop("verbosity", 0),
            "disable_update_check": True,
            **args,
        }
        self.search_config = search_config.copy()
        estimator = (
            TPOTRegressor(**search_config)
            if mode == "regression"
            else TPOTClassifier(**search_config)
        )
        estimator.fit(X_train, y_train)

        self.scores = {}
        for scorer_name in self.scorer_names:
            scorer = get_scorer(scorer_name)
            self.scores[scorer_name] = float(
                scorer(estimator, X_validation, y_validation)
            )

        self.model = estimator.fitted_pipeline_
        self.save()
        return ""

    def predict(self, df: DataFrame, args=None) -> DataFrame:
        if self.model is None:
            raise RuntimeError("model is not loaded; call train() or load() first")

        missing = [column for column in self.input_features if column not in df]
        if missing:
            raise ValueError(f"missing input columns: {missing}")
        predict_data = self._encode_categories(df[self.input_features].copy())
        predictions = self.model.predict(predict_data)
        output_df = df.copy()
        output_df["prediction"] = predictions
        if hasattr(self.model, "predict_proba"):
            output_df["probability"] = self.model.predict_proba(predict_data)[
                :, 1
            ]
        return output_df

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "task_type": self.task_type,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "scorer_names": self.scorer_names,
            "scores": self.scores,
            "category_mappings": self.category_mappings,
            "search_config": self.search_config,
            "random_state": self.random_state,
        }
        with self.model_path.open("wb") as artifact:
            pickle.dump(bundle, artifact)

    def load(self) -> None:
        with self.model_path.open("rb") as artifact:
            bundle = pickle.load(artifact)
        self.model = bundle["model"]
        self.task_type = bundle["task_type"]
        self.input_features = bundle["input_features"]
        self.output_features = bundle["output_features"]
        self.scorer_names = bundle["scorer_names"]
        self.scores = bundle["scores"]
        self.category_mappings = bundle["category_mappings"]
        self.search_config = bundle["search_config"]
        self.random_state = bundle["random_state"]

    def describe(self, attribute: str | None = None):
        if attribute == "info":
            description = {
                "task_type": [self.task_type],
                "input": [json.dumps(self.input_features)],
                "output": [json.dumps(self.output_features)],
                "score_names": [json.dumps(self.scorer_names)],
                "scores": [json.dumps(self.scores)],
                "search_config": [json.dumps(self.search_config)],
            }
            return DataFrame.from_dict(description)
        return DataFrame(["info"], columns=["tables"])

    def _fit_category_mappings(
        self, frame: DataFrame
    ) -> dict[str, dict[Any, int]]:
        mappings = {}
        for column in frame.select_dtypes(include=["object", "category"]).columns:
            values = frame[column].value_counts().sort_values(
                ascending=False, kind="stable"
            )
            mappings[column] = {
                value: index for index, value in enumerate(values.index)
            }
        return mappings

    def _encode_categories(self, frame: DataFrame) -> DataFrame:
        encoded = frame.copy()
        for column, mapping in self.category_mappings.items():
            encoded[column] = (
                encoded[column]
                .astype(object)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
        return encoded
