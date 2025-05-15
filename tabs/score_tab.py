import streamlit as st
import os
from logic.predict_score import predict_score
import pandas as pd

def get_model_list(model_dir="models"):
    return sorted([
        f for f in os.listdir(model_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(model_dir, f))
    ])

def show():
    st.subheader("⚡ 高設定スコアを出力（XGBoost）")
    st.markdown("事前に前処理とモデル学習を済ませておく必要があります。")

    prepared_path = "output/prepared_data.csv"
    model_dir = "models"
    output_path = "output/predicted_with_score.csv"

    # ファイルチェック
    if not os.path.exists(prepared_path):
        st.error(f"❌ `{prepared_path}` が存在しません。先に前処理を実行してください。")
        return

    model_list = get_model_list(model_dir)
    if not model_list:
        st.error(f"❌ `{model_dir}` に .json モデルが見つかりません。先にモデルを学習・保存してください。")
        return

    selected_model = st.selectbox("使用するモデルを選択", model_list)
    model_path = os.path.join(model_dir, selected_model)

    if st.button("⚡ スコア予測を実行"):
        with st.spinner(f"{selected_model} を使ってスコアを計算中..."):
            try:
                df = predict_score(prepared_path, model_path, output_path)
                st.success(f"✅ スコア予測完了！出力: `{output_path}`")
                st.dataframe(df[["ホール名", "機種名", "台番号", "高設定予測スコア"]].sort_values(
                    "高設定予測スコア", ascending=False).head(20), use_container_width=True)
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {e}")
