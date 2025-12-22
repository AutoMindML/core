import os
import pickle
from typing import Optional

from pandas import DataFrame
from tpot import TPOTClassifier, TPOTRegressor

# refs:
# https://docs.mindsdb.com/integrations/ai-engines/byom
# https://github.com/EpistasisLab/tpot
# https://epistasislab.github.io/tpot/latest/tpot_api/classifier/


class TPOTEngine:
    def __init__(self):
        self.model = None
        self.model_path = "tpot_best_pipeline.pkl"

    def train(self, df: DataFrame, target_col: str, args=None):
        # Setup default args
        args = args if args else {}
        mode = args.get("mode", "classification").lower()
        max_time_mins = args.get("max_time_mins", 5)

        # Data preparation
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # simple transformation to ensure model be able to run
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes

        # Initialize automl engine
        tpot = None
        if mode == "regression":
            tpot = TPOTRegressor(max_time_mins=max_time_mins)
        else:
            tpot = TPOTClassifier(max_time_mins=max_time_mins)

        tpot.fit(X, y)

        self.model = tpot.fitted_pipeline_

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

    def describe(self, model_state, attribute: Optional[str] = None):
        if attribute == "info":
            return DataFrame()
        else:
            tables = ["info"]
            return DataFrame(tables, columns=["tables"])
