
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from analysis.forecasting import (
    forecast_machine_with_prophet,
    batch_forecast_for_hall,
    batch_forecast_all,
)
from utils import sanitize_filename

st.set_page_config(layout="wide", page_title="差枚予測ダッシュボード")

st.title("🎰 Prophetによる差枚予測ダッシュボード")

# アップロード
uploaded_file = st.file_uploader("CSVファイルを選択", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["日付"] = pd.to_datetime(df["日付"])
    halls = sorted(df["ホール名"].dropna().unique().tolist())

    selected_hall = st.selectbox("ホールを選択", halls)

    if selected_hall:
        latest_date = df[df["ホール名"] == selected_hall]["日付"].max()
        filtered = df[(df["ホール名"] == selected_hall) & (df["日付"] == latest_date)]
        machine_counts = (
            filtered.groupby("機種名")["台番号"]
            .nunique()
            .reset_index(name="台数")
            .sort_values("台数", ascending=False)
        )
        machine_options = [f"{row['機種名']}（{row['台数']}台）" for _, row in machine_counts.iterrows()]
        selected_machine = st.selectbox("機種を選択", machine_options)
        forecast_days = st.slider("予測日数", 3, 14, 7)

        if st.button("📈 この機種を予測実行"):
            cleaned_machine = selected_machine.split("（")[0]
            fig = forecast_machine_with_prophet(df[df["ホール名"] == selected_hall], cleaned_machine, forecast_days)
            st.image(fig)

        if st.button("📊 このホール全機種を一括予測"):
            logs = batch_forecast_for_hall(df, selected_hall, forecast_days)
            st.text_area("実行ログ", logs, height=300)

    if st.button("🔁 全ホール一括予測"):
        logs = batch_forecast_all(df, 7)
        st.text_area("全ホールログ", logs, height=300)

    # 最新ログ表示
    log_dir = "output/logs"
    if os.path.exists(log_dir):
        logs = sorted(os.listdir(log_dir), reverse=True)
        if logs:
            latest_log = logs[0]
            st.subheader(f"📄 最新ログ: {latest_log}")
            with open(os.path.join(log_dir, latest_log), encoding="utf-8") as f:
                st.text(f.read())
