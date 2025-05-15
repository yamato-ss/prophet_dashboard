import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import streamlit as st
from sklearn.metrics import roc_auc_score

def self_training_tab():
    st.subheader("🔁 XGBoost 自己学習・評価・保存ループ")

    # ==== UI: 除外期間指定 ====
    st.subheader("📆 除外期間の設定")
    exclude_days = st.slider("再学習に使用しない直近の日数（過去◯日）", min_value=7, max_value=60, value=30)

    # ==== データ読み込み・分割 ====
    df = pd.read_csv("output/prepared_with_score.csv")
    df["日付"] = pd.to_datetime(df["日付"])
    cutoff = pd.to_datetime("today").normalize() - pd.Timedelta(days=exclude_days)

    train_df = df[df["日付"] < cutoff].copy()
    eval_df = df[df["日付"] >= cutoff].copy()

    # ==== データ量チェック ====
    if train_df["日付"].nunique() < 90:
        st.error("⚠ 除外後の学習データが90日未満のため、再学習は実行できません。除外期間を短くしてください。")
        st.stop()

    # ==== 学習・ラベル作成 ====
    train_df["label"] = ((train_df["G数"] > 6000) & (train_df["差枚"] > train_df["G数"] * 3 * 1.05)).astype(int)
    feature_cols = [c for c in train_df.columns if c not in ["ホール名", "機種名", "台番号", "日付", "差枚", "label", "予測スコア"]]
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["label"])
    params = {"objective": "binary:logistic", "eval_metric": "auc"}
    booster = xgb.train(params, dtrain, num_boost_round=100)

    # ==== 評価データで予測・評価 ====
    eval_df["label"] = ((eval_df["G数"] > 6000) & (eval_df["差枚"] > eval_df["G数"] * 3 * 1.05)).astype(int)
    deval = xgb.DMatrix(eval_df[feature_cols])
    eval_df["予測スコア"] = booster.predict(deval)

    auc = roc_auc_score(eval_df["label"], eval_df["予測スコア"])
    st.metric("AUC（検証データ）", f"{auc:.4f}")

    # ==== Precision@K 判定 ====
    def calculate_precision_at_k(df_eval, k=20):
        df_sorted = df_eval.sort_values("予測スコア", ascending=False).head(k)
        df_sorted["的中"] = ((df_sorted["G数"] > 6000) & (df_sorted["差枚"] > df_sorted["G数"] * 3 * 1.05))
        return df_sorted["的中"].mean()

    precision_at_20 = calculate_precision_at_k(eval_df, k=20)
    st.metric("Precision@20", f"{precision_at_20*100:.1f}%")

    # ==== 精度評価の表示 ====
    def render_precision_feedback(precision_at_k, k=20):
        if precision_at_k < 0.3:
            color = "#cc4444"; icon = "⚠️"; label = "危険"
            msg = "精度が著しく低く、予測の信頼性が極めて低い状態です。モデルの再学習を強く推奨します。"
        elif precision_at_k < 0.5:
            color = "#d28e00"; icon = "⚠️"; label = "要注意"
            msg = "予測精度がやや低下しています。設定傾向に変化がある可能性もあります。"
        elif precision_at_k < 0.7:
            color = "#228866"; icon = "✅"; label = "安定"
            msg = "標準的な精度が維持されています。予測は信頼に足ります。"
        else:
            color = "#22aa22"; icon = "🟢"; label = "良好"
            msg = "非常に高精度な予測ができています。狙い台として有望です。"

        st.markdown(f"""
        <div style='border-left: 6px solid {color}; padding: 0.5em 1em; margin-top:1em; color: #333; background-color: #f9f9f9;'>
        <strong>{icon} Precision@{k} 判定: <span style='color:{color}'>{label}</span></strong><br>
        {msg}<br>
        <span style='font-size: 0.9em; color: #666;'>（上位{k}件中の高設定的中率: {precision_at_k*100:.1f}%）</span>
        </div>
        """, unsafe_allow_html=True)

    render_precision_feedback(precision_at_20)

    # ==== 精度履歴に追加・保存 ====
    os.makedirs("log", exist_ok=True)
    log_path = "log/precision_log.csv"
    today_str = datetime.today().strftime("%Y-%m-%d")

    log_entry = pd.DataFrame([{"日付": today_str, "AUC": auc, "Precision@20": precision_at_20}])
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, log_entry], ignore_index=True)
    else:
        log_df = log_entry
    log_df.to_csv(log_path, index=False)

    # ==== 精度変動グラフ表示 ====
    st.subheader("📈 過去の精度変動（Precision@20）")
    fig, ax = plt.subplots()
    ax.plot(log_df["日付"], log_df["Precision@20"], marker="o")
    ax.axhline(0.5, color="red", linestyle="--", label="閾値（0.5）")
    ax.set_title("Precision@20 の変動")
    ax.set_ylabel("的中率")
    ax.set_xlabel("日付")
    ax.legend()
    st.pyplot(fig)

    # ==== モデル保存判断 ====
    st.subheader("💾 モデル保存")
    if precision_at_20 < 0.5:
        save_decision = st.radio("今回の精度では推奨されません。モデルを保存しますか？", ["保存しない", "保存する"], index=0)
    else:
        save_decision = st.radio("モデルを保存しますか？", ["保存する", "保存しない"], index=0)

    if save_decision == "保存する":
        version_name = st.text_input("保存するモデルのバージョン名（例: 2025-05-16_イベント明け）", value="latest")
        if version_name.strip() == "":
            version_name = "latest"

        version_filename = f"xgb_model_{version_name.replace(' ', '_')}.json"
        save_path = os.path.join("models", version_filename)
        os.makedirs("models", exist_ok=True)
        booster.save_model(save_path)
        booster.save_model("models/xgb_model.json")  # 常に最新としても保存
        st.success(f"モデルを保存しました: {save_path}")
    else:
        st.info("モデルは保存されませんでした")
