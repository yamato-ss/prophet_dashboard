import os
import pandas as pd
import glob
from utils.split_by_hall import split_prepared_data_by_hall

def run_preprocessing(input_dir="data", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # ▼ CSV読み込み
    all_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    if not all_files:
        raise FileNotFoundError(f"{input_dir} にCSVファイルが見つかりません。")

    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df["ホール名"] = os.path.basename(os.path.dirname(file))
            df["ファイル名"] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"[⚠️ 読み込み失敗] {file}:", e)

    if not dfs:
        raise ValueError("有効なCSVファイルがありませんでした。")

    merged = pd.concat(dfs, ignore_index=True)

    # 日付抽出
    merged["日付"] = pd.to_datetime(merged["ファイル名"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")

    # 数値変換と空欄補完
    numeric_cols = ["G数", "差枚", "BB", "RB", "合成確率", "BB確率", "RB確率", "ART", "ART確率"]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.replace(",", "").str.replace("+", "").str.replace("−", "-")
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged["台番号"] = merged["台番号"].astype(str)
    merged["末尾"] = merged["台番号"].str[-1].astype(int)
    merged["曜日"] = merged["日付"].dt.weekday
    merged["G数"] = pd.to_numeric(merged["G数"], errors="coerce")
    merged["差枚"] = pd.to_numeric(merged["差枚"], errors="coerce")
    merged["スコア"] = merged["差枚"] + merged["G数"] * 2

    merged["高設定判定"] = (merged["差枚"] > (merged["G数"] * 3 * 1.05) - (merged["G数"] * 3 )) & (merged["G数"] > 6000)

    # 前日差枚
    merged = merged.sort_values(["ホール名", "機種名", "台番号", "日付"])
    merged["前日差枚"] = merged.groupby(["ホール名", "機種名", "台番号"])["差枚"].shift(1).fillna(0)

    # 周期日数（1000枚超え時）
    merged["周期日数"] = (
        merged[merged["差枚"] > 1000]
        .groupby(["ホール名", "機種名", "台番号"])["日付"]
        .diff()
        .dt.days
    )
    merged["周期日数"] = merged["周期日数"].fillna(999)

    # 前日高設定
    merged["前日高設定"] = merged.groupby(["ホール名", "機種名", "台番号"])["高設定判定"].shift(1).fillna(0).astype(int)

    # 欠損除去
    df_clean = merged.dropna(subset=["日付", "台番号", "機種名"])

    # 出力（予測スコアなし）
    prepared_path = os.path.join(output_dir, "prepared_data.csv")
    df_clean.to_csv(prepared_path, index=False)

    # ▼ ホール別に分割保存（prepared_data_for_xgb_train.csv）
    split_prepared_data_by_hall(
        input_path=prepared_path,
        output_root=os.path.join(output_dir, "for_xgb")
    )

    return df_clean
