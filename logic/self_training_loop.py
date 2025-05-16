
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
from logic.train_utils import train_model, save_model, load_prepared_data, load_latest_model
from sklearn.metrics import roc_auc_score
import json

def calculate_auc(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)

def self_train_loop(hall_name, exclude_days=30):
    print(f"[INFO] Hall: {hall_name}")
    input_path = f"output/for_xgb/{hall_name}/prepared_data_for_xgb_train.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} が見つかりません")

    df = pd.read_csv(input_path)
    if "日付" not in df.columns:
        raise ValueError("データに '日付' 列が含まれていません")

    df["日付"] = pd.to_datetime(df["日付"])
    cutoff = df["日付"].max() - timedelta(days=exclude_days)
    train_df = df[df["日付"] < cutoff]
    test_df = df[df["日付"] >= cutoff]

    if len(train_df) < 10000:
        raise ValueError("学習データが1万件未満です。期間を短くしてください。")

    print(f"[INFO] 学習件数: {len(train_df)} / 検証件数: {len(test_df)}")

    model_new, auc_new, report, used_features = train_model(train_df)
    print(f"[NEW] AUC: {auc_new:.4f}")

    model_dir = f"models/{hall_name}"
    os.makedirs(model_dir, exist_ok=True)

    model_old, path_old = load_latest_model(model_dir)
    auc_old = 0

    if model_old is not None:
        try:
            test_df_cleaned = test_df.drop(columns=[col for col in ["機種名", "ホール名", "ファイル名", "日付"] if col in test_df.columns], errors="ignore")
            X_test = test_df_cleaned[used_features]
            y_test = test_df_cleaned["高設定判定"]
            auc_old = calculate_auc(model_old, X_test, y_test)
            print(f"[OLD] {os.path.basename(path_old)} AUC: {auc_old:.4f}")
        except Exception as e:
            print(f"[WARN] 既存モデルの評価に失敗: {e}")

    if auc_new > auc_old:
        save_name = f"xgb_model_{datetime.now().strftime('%Y%m%d')}.json"
        save_path = os.path.join(model_dir, save_name)
        model_new.save_model(save_path)
        print(f"[SAVE] {save_path} にモデルを保存しました（AUC {auc_old:.4f} → {auc_new:.4f}）")

        # 特徴量保存
        features_name = f"{save_name}_features.json"
        features_path = os.path.join(model_dir, features_name)
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(used_features, f, ensure_ascii=False, indent=2)
        model_saved = True
    else:
        print("[SKIP] 精度が向上していないため保存しません")
        model_saved = False

    log = {
        "hall": hall_name,
        "date": str(datetime.now()),
        "exclude_days": exclude_days,
        "auc_old": auc_old,
        "auc_new": auc_new,
        "model_saved": model_saved
    }

    log_dir = f"logs/{hall_name}"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/self_train_log_{datetime.now():%Y%m%d_%H%M%S}.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hall-name", type=str, required=True, help="対象ホール名")
    parser.add_argument("--exclude-days", type=int, default=30, help="除外する直近の日数")
    args = parser.parse_args()

    self_train_loop(hall_name=args.hall_name, exclude_days=args.exclude_days)
