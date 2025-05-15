import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def train_model(input_path: str, output_model_path: str):
    df = pd.read_csv(input_path)

    features = ["差枚", "G数", "BB", "RB", "ART", "末尾", "曜日", "前日差枚", "周期日数", "前日高設定"]
    X = df[features]
    y = df["高設定判定"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]

    # ⚠️ AUCスコアの安全確認
    if len(set(y_valid)) < 2:
        auc = None
        print("⚠️ ROC AUC score はクラス数が不足のため算出不能")
    else:
        auc = roc_auc_score(y_valid, y_pred_proba)

    model.save_model(output_model_path)

    importance = model.feature_importances_
    df_importance = pd.DataFrame({
        "特徴量": features,
        "重要度": importance
    }).sort_values(by="重要度", ascending=False)

    return df_importance, auc
