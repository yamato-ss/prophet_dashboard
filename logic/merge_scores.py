import pandas as pd

def merge_scores(xgb_df, prophet_score_df, score_type="Cスコア（3日平均）", and_weight=1.0):
    """
    XGBoostとProphetのスコアをAND条件で結合し、スコア列を追加する

    Parameters:
    - xgb_df: 台番号×XGBスコア（predicted_with_score.csv 読込後）
    - prophet_score_df: 機種×Prophetスコア（load_prophet_scores() 出力）
    - score_type: "Bスコア（翌日）" or "Cスコア（3日平均）"
    - and_weight: Prophetスコアに掛ける重み（デフォルト1.0）

    Returns:
    - 台番ごとの ANDスコア付き DataFrame
    """
    # Prophet側のスコア列を選択して整形
    prophet_df = prophet_score_df[["ホール名", "機種名", score_type]].copy()
    prophet_df = prophet_df.rename(columns={score_type: "Prophetスコア"})

    # 台番単位のxgb_dfと機種単位のProphetをjoin（ホール名＋機種名で）
    merged = pd.merge(xgb_df, prophet_df, on=["ホール名", "機種名"], how="left")

    # ANDスコア（Prophetスコア×XGBスコア）を作成
    merged["ANDスコア"] = merged["Prophetスコア"] * merged["高設定予測スコア"] * and_weight

    return merged
