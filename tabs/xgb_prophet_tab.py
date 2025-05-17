import streamlit as st
import pandas as pd
import os
from logic.prophet_score import load_prophet_scores
from logic.merge_scores import merge_scores


def score_tab():
    st.title("🎯 Prophet × XGBoost ANDスコア 狙い台表示")

    st.markdown("""
    このタブでは、Prophetによる未来差枚予測とXGBoostによる高設定スコアを統合し、
    指定した日付における狙い台ランキングを表示します。
    また、選択された日付に対応する予測実行ログも確認できます。

    ※ Prophetの予測データは学習期間と未来予測期間を分けて保存されています。
    """)

    base_dir = "output/for_xgb"
    hall_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    selected_hall = st.selectbox("ホールを選択", hall_names)

    # スコア読み込み（Prophet日付別）
    prophet_df = load_prophet_scores()
    print(prophet_df["対象日"])
    prophet_df["対象日"] = pd.to_datetime(prophet_df["対象日"])
    date_options = sorted(prophet_df["対象日"].unique(), reverse=True)
    date_str_options = [d.strftime("%Y-%m-%d") for d in date_options]
    selected_date_str = st.selectbox("スコア対象日を選択", date_str_options)
    selected_date = pd.to_datetime(selected_date_str)

    # XGB読み込み
    xgb_path = os.path.join(base_dir, selected_hall, "predicted_with_score.csv")
    try:
        xgb_df = pd.read_csv(xgb_path)
    except FileNotFoundError:
        st.error(f"{xgb_path} が見つかりません")
        return

    # 直近3日分の平均スコアを使う
    xgb_df["日付"] = pd.to_datetime(xgb_df["日付"])
    latest_dates = xgb_df["日付"].dropna().sort_values().unique()[-3:]
    xgb_recent = xgb_df[xgb_df["日付"].isin(latest_dates)]
    xgb_df = (
        xgb_recent.groupby(["ホール名", "機種名", "台番号"])[["高設定予測スコア"]]
        .mean()
        .reset_index()
    )
    # 対象日だけのスコアに絞る
    prophet_day_df = prophet_df[prophet_df["対象日"] == selected_date].copy()
    prophet_day_df = prophet_day_df.rename(columns={"Prophetスコア（yhat）": "Prophetスコア"})

    # 結合・スコア計算
    merged = merge_scores(xgb_df, prophet_day_df, score_type="Prophetスコア")
    merged = merged.sort_values("ANDスコア", ascending=False)
    merged = merged.drop_duplicates(subset=["ホール名", "機種名", "台番号"], keep="first")
    min_score = st.number_input("Prophetスコアの下限", value=0.0)
    top_n = st.slider("表示件数（上位台数）", 10, 100, 30)
    filtered = merged[(merged["Prophetスコア"] >= min_score)].copy()
    ranked = filtered.sort_values("ANDスコア", ascending=False).head(top_n)

    # 表示
    st.markdown(f"### 📅 {selected_date_str} 時点のANDスコア上位 {top_n} 台")
    st.dataframe(ranked[["ホール名", "機種名", "台番号", "Prophetスコア", "高設定予測スコア", "ANDスコア"]])

    # ログ表示
    if st.checkbox("📄 この日の予測実行ログを表示"):
        log_path = f"output/logs/{selected_date_str}.log"
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                st.text_area("実行ログ", f.read(), height=300)
        else:
            st.warning("この日付のログファイルは存在しません。")

    # ダウンロード
    csv = ranked.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 この結果をCSVで保存", csv, file_name="and_score_result.csv", mime="text/csv")
