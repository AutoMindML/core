import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automind.console import console
from automind.data.csv.file import AvailableDatasetsCSV
from automind.data.main import load_data

# 讀取資料
df = load_data(AvailableDatasetsCSV.diabetes.name)

# 將 0 視為缺失值的欄位
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

# 分割特徵與目標
X = df.drop(columns="Outcome")
y = df["Outcome"]

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 建立 Pipeline：缺失值處理 + 標準化 + 模型
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# 交叉驗證評估
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

# 訓練模型
pipeline.fit(X_train, y_train)

# 測試集預測
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# 評估指標
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 畫出 ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# plt.figure()
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("/mnt/data/roc_curve.png")

# 儲存模型
# joblib.dump(pipeline, "/mnt/data/diabetes_model.pkl")

# 輸出結果
console.print(
    {
        "cross_val_auc": cv_scores.mean(),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": roc_auc,
    }
)
