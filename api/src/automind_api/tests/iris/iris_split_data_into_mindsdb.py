from pandas import DataFrame
from sklearn.model_selection import train_test_split

from ...db.connection import connect_mindsdb_server

if __name__ == "__main__":
    mindsdb_server = connect_mindsdb_server()
    iris_df = mindsdb_server.get_database("files").get_table("iris").fetch() # pyright: ignore
    iris_df.drop(columns="Id", inplace=True)
    train_df, test_df = train_test_split(iris_df, test_size=0.1, random_state=42)

    train_df = DataFrame(train_df)
    test_df = DataFrame(test_df)

    files_db = mindsdb_server.get_database("files") # pyright: ignore
    files_db.create_table("iris_train", train_df)
    files_db.create_table("iris_test", test_df)
