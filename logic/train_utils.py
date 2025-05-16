
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os
import json
from datetime import datetime
import glob
import tempfile
import shutil

def load_prepared_data(filepath="output/prepared_for_xgb.csv"):
    df = pd.read_csv(filepath)
    if "高設定判定" not in df.columns:
        raise ValueError("データに '高設定判定' 列が含まれていません。")
    return df

def train_model(df, target_col="高設定判定", test_size=0.2, random_state=42):
    drop_cols = ["機種名", "ホール名", "ファイル名", "日付"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    features = [col for col in df.columns if col != target_col]
    X = df[features]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    report = classification_report(y_test, y_pred_proba > 0.5, output_dict=True)

    return model, auc, report, features

def save_model(model, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_{datetime.now():%Y%m%d_%H%M%S}.json")
    model.save_model(model_path)
    return model_path

def load_latest_model(models_dir):
    json_files = sorted(glob.glob(os.path.join(models_dir, "model_*.json")), reverse=True)
    if not json_files:
        return None, None

    original_path = json_files[0]
    temp_dir = tempfile.mkdtemp(prefix="xgb_temp_")
    temp_model_path = os.path.join(temp_dir, "temp_model.json")
    shutil.copyfile(original_path, temp_model_path)

    model = xgb.XGBClassifier()
    model.load_model(temp_model_path)

    return model, original_path
