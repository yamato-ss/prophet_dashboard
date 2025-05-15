import streamlit as st
import pandas as pd
import os


def prophet_tab():
    st.subheader("🔮 機種別・差枚の未来予測（Prophet）")

    st.markdown("### 🏢 ホールと機種の選択")
    st.markdown("最新日付に存在する機種のみを対象に、Prophetで未来予測を行います。")

    # ✅ 前処理成功時の自動再読み込み通知
    if st.session_state.get("preprocessing_success", False):
        st.info("✅ 前処理が完了したため、最新データを読み込みました。")
        st.session_state["preprocessing_success"] = False  # 一度だけ表示

    try:
        df = pd.read_csv("output/prepared_data.csv")
        print(df)
        df["日付"] = pd.to_datetime(df["日付"])

        halls = sorted(df["ホール名"].dropna().unique().tolist())
        selected_hall = st.selectbox("ホールを選択", halls, key="hall_prophet")

        if selected_hall:
            latest_date = df[df["ホール名"] == selected_hall]["日付"].max()
            latest_models = (
                df[(df["ホール名"] == selected_hall) & (df["日付"] == latest_date)]["機種名"]
                .dropna()
                .value_counts()
                .sort_values(ascending=False)
            )
            models_with_counts = [f"{k}（{v}台）" for k, v in latest_models.items()]
            selected_model = st.selectbox("機種を選択", models_with_counts)

            if st.button("📈 予測を実行する"):
                st.info(f"✅ {selected_hall} - {selected_model} の予測を実行（仮）")
                # 実処理呼び出し部分をここに統合予定
    except FileNotFoundError:
        st.warning("output/prepared_data.csv が見つかりません。データ統合・特徴量生成を実行してください。")
        return
