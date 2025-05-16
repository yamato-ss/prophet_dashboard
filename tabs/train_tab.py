import streamlit as st
import os
import glob
from logic.train_xgb_model import train_model

def show():
    st.header("🏫 XGBoostモデル学習（ホール別）")

    # 📂 ホール一覧の自動取得
    hall_dirs = glob.glob("output/for_xgb/*/")
    hall_names = [os.path.basename(os.path.normpath(d)) for d in hall_dirs]

    if not hall_names:
        st.error("output/for_xgb/ にホール別データが存在しません。前処理を確認してください。")
        return

    selected_hall = st.selectbox("ホールを選択", hall_names)

    # 📄 対象CSVパス
    csv_path = f"output/for_xgb/{selected_hall}/prepared_data_for_xgb_train.csv"

    if not os.path.exists(csv_path):
        st.error(f"{csv_path} が見つかりません。")
        return
    else:
        st.success(f"学習対象ファイル: {csv_path}")

    # 💾 モデル名入力（拡張子なし）
    model_name = st.text_input("保存するモデル名（例: xgb_model_202505）", value="xgb_model_202505")
    model_output_dir = f"models/{selected_hall}"
    os.makedirs(model_output_dir, exist_ok=True)
    output_model_path = f"{model_output_dir}/{model_name}.json"

    # 🟢 学習実行
    if st.button("学習を実行"):
        with st.spinner("モデルを学習中..."):
            try:
                df_importance, auc = train_model(csv_path, output_model_path)
                st.success(f"✅ モデル保存完了: {output_model_path}")
                if auc is not None:
                    st.metric("AUC", round(auc, 4))
                else:
                    st.warning("ROC AUCはクラスの偏りにより算出できませんでした。")

                st.subheader("📊 特徴量重要度")
                st.dataframe(df_importance, use_container_width=True)
            except Exception as e:
                st.error(f"学習中にエラーが発生しました: {e}")
