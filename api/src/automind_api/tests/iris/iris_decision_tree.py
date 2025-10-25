from pandas import DataFrame
from sklearn import tree


class IrisPredictor:
    def train(self, df: DataFrame, target_col: str):
        self.target_col = target_col
        self.model = tree.DecisionTreeClassifier()

        y = df[self.target_col]
        x = df.drop(columns=[self.target_col])
        self.x_cols = list(x.columns)

        self.predictor = self.model.fit(x, y)

    def predict(self, df: DataFrame):
        x = df[self.x_cols]
        res = self.predictor.predict(x)

        pred_df = DataFrame(res, columns=[self.target_col])

        return pred_df
