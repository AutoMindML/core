import json
import os
import pickle
from typing import List, Optional

from pandas import DataFrame
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from tpot.tpot import TPOTClassifier, TPOTRegressor

# refs:
# https://docs.mindsdb.com/integrations/ai-engines/byom
# https://github.com/EpistasisLab/tpot
# https://epistasislab.github.io/tpot/latest/tpot_api/classifier/


class TPOTEngine:
    def __init__(self):
        self.model = None
        self.task_type = None
        self.classification_scorer_names = []
        self.regression_scorer_names = []
        self.input_features: List[str] = []
        self.output_features: List[str] = []
        self.model_path = "tpot_best_pipeline.pkl"
        self.scorer_names: List[str] = []
        self.scores = {}

    def train(self, df: DataFrame, target_col: str, args=None):
        # Setup default args
        args = args if args else {}
        mode = args.get("mode", "classification").lower()
        max_time_mins = args.get("max_time_mins", 5)

        self.task_type = mode

        # https://scikit-learn.org/stable/modules/model_evaluation.html#string-name-scorers
        self.classification_scorer_names = args.get(
            "classification_scorers", "accuracy,f1"
        ).split(",")
        self.regression_scorerss = args.get(
            "regression_scorers", "neg_root_mean_squared_error"
        ).split(",")

        # Data preparation
        y = df[target_col]
        X = df.drop(columns=[target_col])

        self.input_features = X.columns.tolist()
        self.output_features = [target_col]

        # simple transformation to ensure model be able to run
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes

        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, train_size=0.75, test_size=0.25
        )

        # Initialize automl engine
        est = None

        if self.task_type == "regression":
            est = TPOTRegressor(max_time_mins=max_time_mins)
            self.scorer_names = self.regression_scorer_names
        else:
            est = TPOTClassifier(max_time_mins=max_time_mins)
            self.scorer_names = self.classification_scorer_names

        est.fit(X_train, y_train)

        # Validation
        for scorer_name in self.scorer_names:
            scorer = get_scorer(scorer_name)
            score = scorer(est, X_validation, y_validation)
            self.scores[scorer_name] = score

        self.model = est.fitted_pipeline_  # # pyright: ignore

        # Save model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        return ""

    def predict(self, df, args=None):
        if self.model is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            else:
                raise FileNotFoundError("Model not found. Please train first.")

        # predict df must has no target column
        predict_data = df.copy()
        for col in predict_data.select_dtypes(
            include=["object", "category"]
        ).columns:
            predict_data[col] = predict_data[col].astype("category").cat.codes

        predictions = self.model.predict(predict_data)

        output_df = df.copy()
        output_df["prediction"] = predictions

        return output_df

    def describe(self, attribute: Optional[str] = None):
        if attribute == "info":
            description = {
                "task_type": [self.task_type],
                "input": [json.dumps(self.input_features)],
                "output": [json.dumps(self.output_features)],
                "score_names": [json.dumps(self.scorer_names)],
                "scores": [json.dumps(self.scores)],
            }
            return DataFrame.from_dict(description)
        else:
            tables = ["info"]
            return DataFrame(tables, columns=["tables"])
