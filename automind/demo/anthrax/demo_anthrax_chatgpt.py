from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from automind.data.csv.file import AvailableDataset
from automind.data.main import load_data
from automind.evaluation import cross_validation, print_classification_report


def classify_rate(rate):
    if rate <= 33:
        return 0  # 低
    elif rate <= 66:
        return 1  # 中
    else:
        return 2  # 高


target_names = ["低", "中", "高"]


def validation():
    df_train = load_data(AvailableDataset.anthrax_train)

    df_train["發芽等級"] = df_train["發芽率"].apply(classify_rate)

    X = df_train[["時間", "溫度", "濕度"]]
    y = df_train["發芽等級"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    all_y_true, all_y_pred = cross_validation(X, y, model)

    y_pred = model.predict(X_test)
    print_classification_report(y_test, y_pred, "ChatGPT Validation")


def testing():
    df_train = load_data(AvailableDataset.anthrax_train)
    df_test = load_data(AvailableDataset.anthrax_test)

    df_train["發芽等級"] = df_train["發芽率"].apply(classify_rate)
    df_test["發芽等級"] = df_test["發芽率"].apply(classify_rate)

    X_train = df_train[["時間", "溫度", "濕度"]]
    y_train = df_train["發芽等級"]

    X_test = df_test[["時間", "溫度", "濕度"]]
    y_test = df_test["發芽等級"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print_classification_report(y_test, y_pred, "ChatGPT Testing")


if __name__ == "__main__":
    validation()
    testing()
