from pandas import DataFrame
from sklearn.model_selection import train_test_split

from ...db.connection import connect_mindsdb_server
from .iris_decision_tree import IrisPredictor

if __name__ == "__main__":
    mindsdb_server = connect_mindsdb_server()
    iris_df = mindsdb_server.get_database("files").get_table("iris").fetch() # pyright: ignore

    iris_df.drop(columns="Id", inplace=True)

    print(iris_df)

    train_df, test_df = train_test_split(iris_df, test_size=0.2, random_state=42)
    predictor = IrisPredictor()

    train_df = DataFrame(train_df)
    test_df = DataFrame(test_df)

    res = predictor.train(train_df, "Species")
    print(res)

    res = predictor.predict(test_df)
    print(res)
