# XGBoostモデルによる予測スコアの生成処理
import os
import pandas as pd
import xgboost as xgb

def predict_score(
    input_path="output/prepared_data.csv",
    model_path="models/xgb_model.json",
    output_path="output/predicted_with_score.csv"
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"入力ファイルが存在しません: {input_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが存在しません: {model_path}")

    # 前処理済みデータ読み込み
    df = pd.read_csv(input_path)

    # 特徴量（※モデルと一致するもの）
    ordered_feature_cols = [
        "差枚", "G数", "BB", "RB", "ART", "末尾", "曜日", "前日差枚", "周期日数", "前日高設定"
    ]
    for col in ordered_feature_cols:
        if col not in df.columns:
            raise ValueError(f"必要な列が見つかりません: {col}")

    X = df[ordered_feature_cols].astype("float32")

    # モデル読み込み
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 予測スコア（クラス1の確率）
    df["高設定予測スコア"] = model.predict_proba(X)[:, 1]

    # スコア履歴列を追加（並び順前提）
    df = df.sort_values(["ホール名", "機種名", "台番号", "日付"])
    df["前日スコア"] = df.groupby(["ホール名", "機種名", "台番号"])["高設定予測スコア"].shift(1)
    df["前日差枚"] = df.groupby(["ホール名", "機種名", "台番号"])["差枚"].shift(1)

    # 保存
    df.to_csv(output_path, index=False)
    print(f"[✅ スコア予測完了] 保存先: {output_path}")

    return df
