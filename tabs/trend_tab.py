import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from utils.common import get_japanese_font

@st.cache_data
def load_scored_data(path="output/predicted_with_score.csv"):
    if not os.path.exists(path):
        st.error(f"予測スコアファイルが見つかりません: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["日付"])

def show():
    st.subheader("🎯 高設定スコア予測ランキング（XGBoost）")
    st.markdown("""
    このタブでは、XGBoostモデルによって予測された「高設定スコア」をもとに、ホール別・最新日のランキングを表示します。
    """)

    # 🔧 事前案内
    st.markdown("""
    ### ℹ️ 表示の前提

    このランキングは、以下の処理を完了した後に表示できます：

    1. **データ前処理（prepared_data.csv）**
    2. **XGBoostモデルの学習（models/*.json）**
    3. **スコア出力（predicted_with_score.csv）**

    → いずれかが未実行の場合は、右の「⚙️ データ生成・操作」タブから処理を行ってください。
    """)

    df = load_scored_data()
    if df.empty:
        st.warning("予測データがありません。先にスコアを生成してください。")
        return

    # ▼ ホール選択
    hall_list = sorted(df["ホール名"].dropna().unique())
    selected_hall = st.selectbox("ホールを選択してください", hall_list)

    hall_df = df[df["ホール名"] == selected_hall]
    if hall_df.empty:
        st.warning("該当ホールのデータが見つかりません。")
        return

    # ▼ 日付選択（最新順にソート）
    date_list = sorted(hall_df["日付"].dropna().unique(), reverse=True)
    date_str_list = [d.strftime("%Y-%m-%d") for d in date_list]

    # ▼ 表示は文字列、選択後にインデックスでdatetimeへ戻す
    selected_date_str = st.selectbox("対象日付を選択してください", date_str_list)
    selected_date = date_list[date_str_list.index(selected_date_str)]

    # ▼ ランキングデータ抽出（指定ホール・日付）
    latest_df = hall_df[hall_df["日付"] == selected_date].copy()

    st.markdown("---")
    st.markdown("### 📊 読み取りポイントに基づく集計")

    # 🔢 総差枚
    st.metric("🏢 ホール全体の総差枚数", f"{latest_df['差枚'].sum():,} 枚")

    st.markdown("#### 🧷 差枚+1000以上連番クラスタの分布と対象台番（機種名付き）")
    # 🧷 高スコア連番クラスタの分布と対象台番
    latest_df = hall_df[hall_df["日付"] == selected_date].copy()
    ranked_df = latest_df.sort_values("差枚", ascending=False).copy()

    try:
        high_df = latest_df[latest_df["差枚"] >= 1000].copy()
        high_df["台番号_int"] = high_df["台番号"].astype(int)
        sorted_nums = sorted(high_df["台番号_int"].tolist())

        clusters = []
        cluster = [sorted_nums[0]]
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                cluster.append(sorted_nums[i])
            else:
                if len(cluster) > 1:
                    clusters.append(cluster)
                cluster = [sorted_nums[i]]
        if len(cluster) > 1:
            clusters.append(cluster)

        from collections import Counter
        cluster_sizes = [len(c) for c in clusters]
        cluster_summary = Counter(cluster_sizes)
        cluster_df = pd.DataFrame.from_dict(cluster_summary, orient="index").reset_index()
        cluster_df.columns = ["連続台数", "件数"]
        cluster_df = cluster_df.sort_values("連続台数")
        st.dataframe(cluster_df, use_container_width=True)

        # 台番 → 機種名辞書を作成
        num_to_model = dict(zip(latest_df["台番号"].astype(int), latest_df["機種名"]))
        for c in clusters:
            labels = [f"{n}（{num_to_model.get(n, '不明')}）" for n in c]
            st.markdown(f"・{' → '.join(labels)}")

    except Exception as e:
        st.warning(f"並びクラスタ解析でエラーが発生しました: {e}")
    
    # 🔢 台番号末尾の出現傾向（スコア平均 + 差枚平均 + G数平均 + 高設定台数 + 差枚勝率）
    st.markdown("#### 🔢 台番号末尾ごとのスコア・差枚傾向と設定状況")
    try:
        latest_df["末尾"] = latest_df["台番号"].astype(str).str[-1]
        latest_df["高設定判定"] = (latest_df["高設定予測スコア"] > 0.6)
        latest_df["差枚プラス"] = latest_df["差枚"] > 0

        summary = (
            latest_df.groupby("末尾")
            .agg(
                件数=("台番号", "count"),
                平均スコア=("高設定予測スコア", "mean"),
                平均差枚=("差枚", "mean"),
                平均G数=("G数", "mean"),
                高設定台数=("高設定判定", "sum"),
                差枚プラス台数=("差枚プラス", "sum")
            )
            .reset_index()
        )

        summary["平均差枚"] = summary["平均差枚"].round(0).astype(int)
        summary["平均G数"] = summary["平均G数"].round(0).astype(int)
        summary["勝率"] = summary.apply(lambda x: f"{int(x['差枚プラス台数'])}/{x['件数']}", axis=1)
        st.dataframe(summary[["末尾", "平均スコア", "平均差枚", "平均G数", "高設定台数", "勝率"]],
                    use_container_width=True)
    except Exception as e:
        st.warning(f"末尾スコア分析でエラーが発生しました: {e}")

    # 🔁 差枚マイナス台のリベンジ傾向分析
    st.markdown("#### 🔁 前日差枚 0~-2000/-2000~-5000/-5000~ の台のリベンジ傾向分析")

    try:
        rev_df = latest_df[((latest_df["前日差枚"]) < 0 & (latest_df["前日差枚"] >= -2000))].copy()
        total_targets = len(rev_df)
        if total_targets == 0:
            st.info("レンジ: 0~-2000 に該当するリベンジ対象台がありません。")
        else:
            success_df = rev_df[rev_df["差枚"] > 1000]
            success_count = len(success_df)
            win_rate = (success_count / total_targets) * 100

            st.markdown(f"🎯 **レンジ: 0~-2000 リベンジ対象台数**：{total_targets} 台")
            st.markdown(f"✅ **レンジ: 0~-2000 リベンジ成功台数**：{success_count} / {total_targets} 台（{win_rate:.1f}%）")

            # 内訳：成功台の機種別
            breakdown = success_df["機種名"].value_counts().reset_index()
            breakdown.columns = ["機種名", "成功台数"]
            st.markdown("##### ✅ リベンジ成功台の機種内訳")
            st.dataframe(breakdown, use_container_width=True)
        
        rev_df = latest_df[((latest_df["前日差枚"] < -2000) & (latest_df["前日差枚"] > -5000))].copy()
        total_targets = len(rev_df)
        if total_targets == 0:
            st.info("レンジ: -2000~-5000 該当するリベンジ対象台がありません。")
        else:
            success_df = rev_df[rev_df["差枚"] > 1000]
            success_count = len(success_df)
            win_rate = (success_count / total_targets) * 100

            st.markdown(f"🎯 **レンジ: -2000~-5000 リベンジ対象台数**：{total_targets} 台")
            st.markdown(f"✅ **レンジ: -2000~-5000 リベンジ成功台数**：{success_count} / {total_targets} 台（{win_rate:.1f}%）")

            # 内訳：成功台の機種別
            breakdown = success_df["機種名"].value_counts().reset_index()
            breakdown.columns = ["機種名", "成功台数"]
            st.markdown("##### ✅ リベンジ成功台の機種内訳")
            st.dataframe(breakdown, use_container_width=True)
        
        rev_df = latest_df[(latest_df["前日差枚"] < -5000)].copy()
        total_targets = len(rev_df)
        if total_targets == 0:
            st.info("レンジ: -5000~ 該当するリベンジ対象台がありません。")
        else:
            success_df = rev_df[rev_df["差枚"] > 1000]
            success_count = len(success_df)
            win_rate = (success_count / total_targets) * 100

            st.markdown(f"🎯 レンジ: -5000~ **リベンジ対象台数**：{total_targets} 台")
            st.markdown(f"✅ レンジ: -5000~ **リベンジ成功台数**：{success_count} / {total_targets} 台（{win_rate:.1f}%）")

            # 内訳：成功台の機種別
            breakdown = success_df["機種名"].value_counts().reset_index()
            breakdown.columns = ["機種名", "成功台数"]
            st.markdown("##### ✅ リベンジ成功台の機種内訳")
            st.dataframe(breakdown, use_container_width=True)

    except Exception as e:
        st.warning(f"リベンジ傾向分析でエラーが発生しました: {e}")
