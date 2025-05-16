import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json

def train_model(input_path="output/prepared_data.csv", model_path="models/xgb_model.json"):
    df = pd.read_csv(input_path)

    features = [
        "差枚", "G数", "BB", "RB", "ART", "末尾", "曜日",
        "前日差枚", "周期日数", "前日高設定"
    ]
    X = df[features]
    y = df["高設定判定"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"✅ モデルAUC: {auc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # 特徴量保存
    features_path = model_path.replace(".json", "_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    print(f"✅ モデル保存: {model_path}")
    print(f"✅ 特徴量保存: {features_path}")

    return model, auc
