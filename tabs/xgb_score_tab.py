import streamlit as st
import os
import pandas as pd
from logic.predict_score import predict_score

def score_tab():
    st.subheader("⚡ 高設定スコアを出力（ホール別）")
    st.markdown("前処理・ホール別モデル学習が完了している必要があります。")

    hall_root = "output/for_xgb"
    model_root = "models"

    # ▼ ホール選択
    halls = sorted([
        d for d in os.listdir(hall_root)
        if os.path.isdir(os.path.join(hall_root, d))
    ])
    if not halls:
        st.error("❌ ホール別データが見つかりません。前処理を実行してください。")
        return

    selected_hall = st.selectbox("📂 ホールを選択", halls)

    # ▼ モデル選択
    hall_model_dir = os.path.join(model_root, selected_hall)
    if not os.path.exists(hall_model_dir):
        st.error(f"❌ モデルディレクトリが存在しません: {hall_model_dir}")
        return

    model_list = sorted([
        f for f in os.listdir(hall_model_dir)
        if f.endswith(".json") and "_features" not in f
    ])
    if not model_list:
        st.error(f"❌ モデルが見つかりません: {hall_model_dir}")
        return

    selected_model = st.selectbox("🤖 モデルを選択", model_list)

    # ▼ パス設定
    input_path = os.path.join(hall_root, selected_hall, "prepared_data_for_xgb_train.csv")
    output_path = os.path.join(hall_root, selected_hall, "predicted_with_score.csv")
    model_path = os.path.join(hall_model_dir, selected_model)

    if not os.path.exists(input_path):
        st.error(f"❌ 入力ファイルが見つかりません: {input_path}")
        return

    if st.button("⚡ スコア予測を実行"):
        with st.spinner("推論中..."):
            try:
                df = predict_score(input_path, model_path, output_path)
                st.success(f"✅ スコア予測完了！保存先: `{output_path}`")
                st.dataframe(
                    df[["機種名", "台番号", "高設定予測スコア"]]
                    .sort_values("高設定予測スコア", ascending=False)
                    .head(20),
                    use_container_width=True
                )
                with open(output_path, "rb") as f:
                    st.download_button("📥 結果CSVをダウンロード", f, file_name=f"{selected_hall}_predicted_with_score.csv")
            except Exception as e:
                st.error("❌ エラーが発生しました。")
                st.exception(e)
