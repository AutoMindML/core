from os import listdir
from pathlib import Path

from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split

from .connection import connect_mindsdb_server


def insert_data_into_mindsdb():
    mindsdb_server = connect_mindsdb_server()

    data_dir = f"{Path(__file__).parent.parent.absolute()}/data"

    for file_name in listdir(data_dir):
        split_file_name = file_name.split(".")

        if split_file_name[1] == "csv":
            df = read_csv(f"{data_dir}/{file_name}")
            files_db = mindsdb_server.get_database("files")

            if split_file_name[0] not in [
                table.name for table in files_db.list_tables()
            ]:
                files_db.create_table(split_file_name[0], df)

    iris_df = mindsdb_server.get_database("files").get_table("iris").fetch()
    iris_df.drop(columns="Id", inplace=True)
    train_df, test_df = train_test_split(iris_df, test_size=0.1, random_state=42)

    train_df = DataFrame(train_df)
    test_df = DataFrame(test_df)

    files_db = mindsdb_server.get_database("files")
    files_db.create_table("iris_train", train_df)
    files_db.create_table("iris_test", test_df)


if __name__ == "__main__":
    insert_data_into_mindsdb()
