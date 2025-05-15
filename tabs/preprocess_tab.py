import streamlit as st
from utils.preprocessing import run_preprocessing
import os

def show():
    st.header("🧹 データ前処理タブ")

    input_dir = st.text_input("📂 入力元CSVフォルダを指定してください", value="C:\\Users\\forev\\Works\\anaslo\\OriginData")

    if st.button("📦 データを生成する"):
        with st.spinner("前処理を実行中です..."):
            try:
                df = run_preprocessing(input_dir=input_dir, output_dir="output")
                st.success("前処理が完了しました。")
                st.dataframe(df.head())
                st.session_state["preprocessing_success"] = True
            except Exception as e:
                st.error("前処理中にエラーが発生しました。")
                st.session_state["preprocessing_success"] = False
                st.exception(e)

    if os.path.exists("output/prepared_data.csv"):
        st.download_button("📥 ダウンロード（prepared_data.csv）", open("output/prepared_data.csv", "rb"), file_name="prepared_data.csv")