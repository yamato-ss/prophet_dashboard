
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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from combine_and_preprocess import combine_and_preprocess

st.set_page_config(layout="wide", page_title="差枚予測ダッシュボード")

st.title("🎰 ホールデータ AI分析ダッシュボード")

# データ結合と前処理のトリガー
if st.button("🔄 OriginData を結合・前処理実行"):
    with st.spinner("処理中..."):
        try:
            output_dir = "../"
            combine_and_preprocess(output_dir)
            st.success("✅ 結合・前処理が完了しました。prepared_for_xgb.csv が出力されました。")
        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")

# アップロード
uploaded_file = st.file_uploader("CSVファイルを選択", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["日付"] = pd.to_datetime(df["日付"])
    halls = sorted(df["ホール名"].dropna().unique().tolist())
    latest_date = df["日付"].max()
    df["日付"] = pd.to_datetime(df["日付"])
    unique_dates = sorted(df["日付"].dt.date.unique())[::-1]
    
    tab1, tab2, tab3 = st.tabs(["📈 Prophet予測", "🧠 XGBoost予測", "🔍 傾向分析"])

    with tab1:
        st.header("📈 Prophetによる差枚予測")

        selected_hall = st.selectbox("ホールを選択", halls, key="hall_prophet")
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
            selected_machine = st.selectbox("機種を選択", machine_options, key="machine_prophet")
            forecast_days = st.slider("予測日数", 3, 14, 7)

            if st.button("📈 この機種を予測実行"):
                cleaned_machine = selected_machine.split("（")[0]
                fig = forecast_machine_with_prophet(df[df["ホール名"] == selected_hall], cleaned_machine, forecast_days)
                st.image(fig)

            if st.button("📊 このホール全機種を一括予測"):
                logs = batch_forecast_for_hall(df, selected_hall, forecast_days)
                st.text_area("実行ログ", logs, height=300)

            if st.button("🔁 全ホール一括予測"):
                logs = batch_forecast_all(df, forecast_days)
                st.text_area("全ホールログ", logs, height=300)

        log_dir = "output/logs"
        if os.path.exists(log_dir):
            logs = sorted(os.listdir(log_dir), reverse=True)
            if logs:
                latest_log = logs[0]
                st.subheader(f"📄 最新ログ: {latest_log}")
                with open(os.path.join(log_dir, latest_log), encoding="utf-8") as f:
                    st.text(f.read())

    with tab2:
        st.header("🧠 高設定スコア予測（XGBoost）")

        selected_hall = st.selectbox("ホールを選択", halls, key="hall_xgb")
        target_date = st.selectbox("予測対象日", unique_dates, key="date_xgb")
        target_datetime = pd.to_datetime(target_date)

        hall_df = df[df["ホール名"] == selected_hall]

        if st.button("🔮 XGBoost予測実行"):
            pred_target = hall_df[hall_df["日付"] == target_datetime]
            train_data = hall_df[hall_df["日付"] < target_datetime]

            if len(pred_target) == 0 or len(train_data) < 100:
                st.warning("予測対象または学習データが不足しています。")
            else:
                feature_cols = ["G数", "差枚", "BB", "RB", "ART", "末尾", "曜日", "スコア"]
                X = train_data[feature_cols]
                y = train_data["高設定"]
                X_pred = pred_target[feature_cols]

                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                model.fit(X, y)

                pred_probs = model.predict_proba(X_pred)[:, 1]
                pred_target = pred_target.copy()
                pred_target["予測確率"] = pred_probs

                ranking = (
                    pred_target[["機種名", "台番号", "G数", "差枚", "スコア", "予測確率"]]
                    .sort_values("予測確率", ascending=False)
                    .reset_index(drop=True)
                )

                st.subheader(f"🏆 {selected_hall} のその日の出玉による設定信頼度（{target_date.strftime('%Y-%m-%d')}）")
                st.dataframe(ranking.head(100))

    with tab3:
        st.header("🔍 末尾・並びなどの傾向分析（今後実装予定）")
        st.info("このタブでは末尾番号や並びによる傾向を可視化予定です。")
