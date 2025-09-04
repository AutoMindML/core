from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from automind.data.csv.file import AvailableDataset
from automind.data.main import load_data
from automind.evaluation import print_classification_report


# 發芽率轉等級
def categorize_germination(rate):
    if rate < 33:
        return "低"
    elif rate < 66:
        return "中"
    else:
        return "高"


def validation():
    df = load_data(AvailableDataset.anthrax_train)

    df["發芽等級"] = df["發芽率"].apply(categorize_germination)

    X = df[["時間", "溫度", "濕度"]]
    y = df["發芽等級"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print_classification_report(y_test, y_pred, "ChatGPT Advanced Validation")


def testing():
    df_train = load_data(AvailableDataset.anthrax_train)
    df_test = load_data(AvailableDataset.anthrax_test)

    df_train["發芽等級"] = df_train["發芽率"].apply(categorize_germination)
    df_test["發芽等級"] = df_test["發芽率"].apply(categorize_germination)

    X_train = df_train[["時間", "溫度", "濕度"]]
    y_train = df_train["發芽等級"]

    X_test = df_test[["時間", "溫度", "濕度"]]
    y_test = df_test["發芽等級"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(random_state=42, class_weight="balanced")

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print_classification_report(y_test, y_pred, "ChatGPT Advanced Testing")


if __name__ == "__main__":
    validation()
    testing()
