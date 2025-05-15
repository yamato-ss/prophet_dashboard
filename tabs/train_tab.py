import streamlit as st
import os
from logic.train_xgb_model import train_model

def show():
    st.header("📚 XGBoostモデル学習")

    # 📁 入力ファイルのパス確認
    default_input_path = "output/prepared_data.csv"
    if not os.path.exists(default_input_path):
        st.error(f"{default_input_path} が存在しません。前処理を実行してください。")
        return

    st.success(f"学習対象ファイル：{default_input_path}")

    # 💾 モデル名の入力（.json なし）
    model_name = st.text_input("保存するモデル名（例: xgb_model_202505）", value="xgb_model_202505")
    output_model_path = f"models/{model_name}.json"

    # 🟢 学習実行ボタン
    if st.button("学習を実行"):
        with st.spinner("学習中..."):
            try:
                df_importance, auc = train_model(default_input_path, output_model_path)
                st.success(f"✅ モデル保存完了: {output_model_path}")
                st.metric("AUC", round(auc, 4))

                st.subheader("🔍 特徴量の重要度")
                st.dataframe(df_importance, use_container_width=True)
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
