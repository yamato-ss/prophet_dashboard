import os
import pandas as pd
import xgboost as xgb
import json
import tempfile
import shutil

def predict_score(input_path="output/prepared_data.csv", model_path="models/xgb_model.json", output_path="output/predicted_with_score.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"入力ファイルが存在しません: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが存在しません: {model_path}")

    features_path = model_path.replace(".json", "_features.json")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_path}")
    with open(features_path, encoding="utf-8") as f:
        ordered_feature_cols = json.load(f)

    df = pd.read_csv(input_path)
    X = df[ordered_feature_cols].astype("float32")

    model = xgb.XGBClassifier()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        shutil.copy(model_path, tmp.name)
        model.load_model(tmp.name)

    df["高設定予測スコア"] = model.predict_proba(X)[:, 1]

    df = df.sort_values(["ホール名", "機種名", "台番号", "日付"])
    df["前日スコア"] = df.groupby(["ホール名", "機種名", "台番号"])["高設定予測スコア"].shift(1).fillna(0.0)
    df["前日差枚"] = df.groupby(["ホール名", "機種名", "台番号"])["差枚"].shift(1).fillna(0.0)

    df.to_csv(output_path, index=False)
    print(f"[✅ スコア予測完了] 保存先: {output_path}")
    return df
