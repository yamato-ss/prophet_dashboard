import pandas as pd
import os
import glob

def combine_and_preprocess(
    output_dir="./",
    origin_dir="../OriginData",
    merged_csv="merged_all.csv",
    output_csv="prepared_for_xgb.csv"
):
    print(f"🔍 データ結合開始: ディレクトリ = {origin_dir}")

    # OriginData配下の全CSVを取得
    all_csv = glob.glob(os.path.join(origin_dir, "*/*.csv"))
    if not all_csv:
        print("❌ 結合対象のCSVが見つかりません。")
        return

    dataframes = []
    for file in all_csv:
        df = pd.read_csv(file)
        df["ホール名"] = os.path.basename(os.path.dirname(file))
        df["ファイル名"] = os.path.basename(file)
        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)
    merged.to_csv(output_dir+merged_csv, index=False)
    print(f"✅ 結合完了: {merged_csv} に保存")

    print("🔄 前処理開始")

    # 日付抽出
    merged["日付"] = pd.to_datetime(merged["ファイル名"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")

    # 数値変換と空欄補完
    numeric_cols = ["G数", "差枚", "BB", "RB", "合成確率", "BB確率", "RB確率", "ART", "ART確率"]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.replace(",", "").str.replace("+", "").str.replace("−", "-")
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # 台番号も文字列化 → 末尾算出のため
    merged["台番号"] = merged["台番号"].astype(str)
    merged["末尾"] = merged["台番号"].str[-1].astype(int)
    merged["曜日"] = merged["日付"].dt.weekday

    # スコア：差枚 + G数重視
    merged["スコア"] = merged["差枚"] + (merged["G数"] * 2)

    # ペイアウト105%ライン（= 機械割105%のライン）
    required_diff = merged["G数"] * 3 * (1.05 - 1.00)  # ＝G数 * 0.15

    # 高設定判定：十分な稼働 & 機械割超え
    merged["高設定"] = ((merged["差枚"] >= required_diff) & (merged["G数"] > 6000)).astype(int)
    
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

    # 前日高設定フラグ
    merged["前日高設定"] = merged.groupby(["ホール名", "機種名", "台番号"])["高設定"].shift(1).fillna(0).astype(int)

    # 欠損除去
    df_clean = merged.dropna(subset=["日付", "台番号", "機種名"])

    # 保存
    df_clean.to_csv(output_dir+output_csv, index=False)
    print(f"✅ 前処理完了: 出力ファイル = {output_csv}")
    print(f"📦 データ件数: {len(df_clean)} 件")

if __name__ == "__main__":
    combine_and_preprocess()