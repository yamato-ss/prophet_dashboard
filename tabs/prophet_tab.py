import streamlit as st
import pandas as pd
import os
from analysis.forecasting import forecast_machine_with_prophet, batch_forecast_all
from utils.common import load_prepared_data_for_hall

def prophet_tab():
    st.title("📈 Prophetによる未来予測")

    # ホール一覧を取得（output/for_xgb/ 配下のディレクトリ）
    hall_base_dir = "output/for_xgb"
    hall_names = sorted([d for d in os.listdir(hall_base_dir) if os.path.isdir(os.path.join(hall_base_dir, d))])
    if not hall_names:
        st.warning("ホールデータが存在しません（output/for_xgb/ 配下）")
        return

    selected_hall = st.selectbox("ホールを選択", hall_names)

    try:
        df = load_prepared_data_for_hall(selected_hall)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # 最新日付に登場する機種を抽出（＝撤去済みは除外）
    df["日付"] = pd.to_datetime(df["日付"])
    latest_date = df["日付"].max()
    latest_df = df[df["日付"] == latest_date]
    machine_counts = latest_df["機種名"].value_counts().reset_index()
    machine_counts.columns = ["機種名", "台数"]
    machine_list = machine_counts.sort_values("台数", ascending=False)["機種名"].tolist()

    if not machine_list:
        st.warning("このホールに有効な機種データが見つかりません。")
        return

    selected_machine = st.selectbox("機種を選択", machine_list)
    predict_days = st.slider("予測日数", 1, 14, 7)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔮 この機種を予測実行"):
            result_image = forecast_machine_with_prophet(df, selected_machine, days=predict_days)
            if isinstance(result_image, str):
                st.error(result_image)
            else:
                st.image(result_image, caption=f"{selected_machine}の予測結果")

    with col2:
        if st.button("🌀 このホールの全機種を一括予測"):
            log = batch_forecast_all(df, days=predict_days)
            st.text_area("ログ出力", log, height=300)
