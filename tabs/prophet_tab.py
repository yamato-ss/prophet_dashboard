import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from logic.forecasting import forecast_machine_with_prophet, batch_forecast_all
from utils.common import load_prepared_data_for_hall, get_japanese_font

jp_font = get_japanese_font()

def show():
    st.header("📈 Prophetによる未来予測")

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

    df["日付"] = pd.to_datetime(df["日付"])
    latest_date = df["日付"].max()
    latest_df = df[df["日付"] == latest_date]
    machine_counts = latest_df["機種名"].value_counts().reset_index()
    machine_counts.columns = ["機種名", "台数"]
    machine_list = machine_counts.sort_values("台数", ascending=False)["機種名"].tolist()

    if not machine_list:
        st.warning("このホールに有効な機種データが見つかりません。")
        return

    predict_days = st.slider("予測日数", 1, 14, 7)
    mode = st.radio("予測モードを選択", ["🔎 特定機種を予測", "🌀 全機種を一括予測"])

    if mode == "🔎 特定機種を予測":
        selected_machine = st.selectbox("機種を選択", machine_list)
        if st.button("🔮 この機種を予測実行"):
            result = forecast_machine_with_prophet(df, selected_machine, days=predict_days)

            if isinstance(result, str):
                st.error(result)
            else:
                image, model, forecast, train_df = result
                st.image(image, caption=f"{selected_machine}の予測結果")

                st.markdown("### 🔍 特徴量ごとの影響（Prophetの分解図）")
                fig1 = model.plot_components(forecast)
                st.pyplot(fig1)

                st.markdown("### 📊 各特徴量の出現頻度と平均差枚")
                flag_cols = [col for col in train_df.columns if col.startswith("is_")]
                stats = []
                for col in flag_cols:
                    sub = train_df[train_df[col] == 1]
                    stats.append({
                        "特徴量": col,
                        "出現数": len(sub),
                        "平均差枚": sub["y"].mean() if not sub.empty else 0
                    })
                df_stats = pd.DataFrame(stats).sort_values("平均差枚", ascending=False)
                st.dataframe(df_stats)

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.barh(df_stats["特徴量"], df_stats["平均差枚"])
                ax2.set_xlabel("平均差枚", fontproperties=jp_font)
                ax2.set_title("特徴量別の平均差枚", fontproperties=jp_font)
                st.pyplot(fig2)

                st.markdown("### 🧠 特徴量ごとの予測寄与（yhat_components）")
                component_cols = [
                    col for col in forecast.columns
                    if col.startswith("is_") and not col.endswith("_lower") and not col.endswith("_upper")
                ]
                if component_cols:
                    contrib_stats = []
                    for col in component_cols:
                        contrib_stats.append({
                            "特徴量": col,
                            "平均寄与": forecast[col].mean(),
                            "最大寄与": forecast[col].max(),
                            "最小寄与": forecast[col].min()
                        })
                    df_contrib = pd.DataFrame(contrib_stats).sort_values("平均寄与", ascending=False)
                    st.dataframe(df_contrib)

                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                    ax3.barh(df_contrib["特徴量"], df_contrib["平均寄与"])
                    ax3.set_title("平均寄与による特徴量ランキング", fontproperties=jp_font)
                    ax3.set_xlabel("平均寄与", fontproperties=jp_font)
                    st.pyplot(fig3)

            # --- 4. 特徴量別の予測誤差傾向（学習データベース） ---
            st.markdown("### ❗ 特徴量ごとの予測誤差傾向（学習データベース）")

            error_stats = []
            for col in flag_cols:
                sub = train_df[train_df[col] == 1]
                if sub.empty:
                    continue
                try:
                    sub_input = sub[["ds"] + [c for c in sub.columns if c.startswith("is_")]]
                    yhat_df = model.predict(sub_input)
                    merged = pd.concat([sub.reset_index(drop=True), yhat_df[["yhat"]].reset_index(drop=True)], axis=1)

                    # NaNを除外してMAE計算
                    merged = merged.dropna(subset=["y", "yhat"])
                    if not merged.empty:
                        mae = (merged["yhat"] - merged["y"]).abs().mean()
                        error_stats.append({
                            "特徴量": col,
                            "出現数": len(merged),
                            "平均実績": merged["y"].mean(),
                            "平均予測": merged["yhat"].mean(),
                            "平均誤差": (merged["y"] - merged["yhat"]).mean(),
                            "MAE": mae
                        })
                    else:
                        error_stats.append({
                            "特徴量": col,
                            "出現数": 0,
                            "平均実績": None,
                            "平均予測": None,
                            "平均誤差": None,
                            "MAE": None
                        })

                except Exception as e:
                    st.warning(f"{col} の誤差分析中にエラー: {str(e)}")
                    continue

            df_error = pd.DataFrame(error_stats)
            if not df_error.empty:
                df_error = df_error.sort_values("MAE", ascending=True)

                # 表示用フォーマット調整（None→"―"）
                df_error_display = df_error.copy()
                df_error_display = df_error_display.fillna("―")

                st.dataframe(df_error_display)

                fig4, ax4 = plt.subplots(figsize=(8, 6))
                ax4.barh(df_error.dropna()["特徴量"], df_error.dropna()["MAE"])
                ax4.set_title("特徴量別 MAE（予測誤差）", fontproperties=jp_font)
                ax4.set_xlabel("MAE", fontproperties=jp_font)
                st.pyplot(fig4)
            else:
                st.info("誤差分析の結果、表示可能な特徴量がありませんでした。")

    else:  # 🌀 全機種一括予測
        if st.button("🌀 このホールの全機種を一括予測"):
            log = batch_forecast_all(df[df["ホール名"] == selected_hall], days=predict_days)
            st.text_area("ログ出力", log, height=300)
