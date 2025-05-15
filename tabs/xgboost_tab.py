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

    st.markdown(f"🗓 **選択日付**: `{selected_date.strftime('%Y-%m-%d')}`")

    # ▼ ランキングデータ抽出（指定ホール・日付）
    latest_df = hall_df[hall_df["日付"] == selected_date].copy()
    ranked_df = latest_df.sort_values("高設定予測スコア", ascending=False).copy()

    # ▼ 表示列を整理
    display_cols = ["機種名", "台番号", "スコア", "差枚", "G数", "高設定予測スコア"]
    ranked_df = ranked_df[display_cols]

    st.dataframe(ranked_df.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 読み取りポイントに基づく集計")


    # 🔢 総差枚
    st.metric("🏢 ホール全体の総差枚数", f"{latest_df['差枚'].sum():,} 枚")

    # 📈 スコア分布ヒストグラム
    st.markdown("#### 📈 スコア分布（全台）")
    fig, ax = plt.subplots()
    sns.histplot(latest_df["高設定予測スコア"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("高設定予測スコア", fontproperties=get_japanese_font())
    ax.set_ylabel("台数", fontproperties=get_japanese_font())
    ax.set_title("スコア分布", fontproperties=get_japanese_font())
    st.pyplot(fig)


    st.markdown("#### 🧩 高スコア台の機種別台数ランキング（スコア > 0.6）")

    # 高スコア台をフィルタ
    high_score_df = latest_df[latest_df["高設定予測スコア"] > 0.6].copy()

    # 機種ごとに台数を集計 → ラベルに「（n台）」を追加
    machine_counts = high_score_df["機種名"].value_counts()
    machine_labels = [f"{name}（{count}台）" for name, count in zip(machine_counts.index, machine_counts.values)]
    machine_summary_df = pd.DataFrame({
        "機種名（台数）": machine_labels,
        "平均スコア": high_score_df.groupby("機種名")["高設定予測スコア"].mean().values,
        "平均差枚": high_score_df.groupby("機種名")["差枚"].mean().values,
        "平均G数": high_score_df.groupby("機種名")["G数"].mean().values
    }).sort_values("平均スコア", ascending=False)

    st.dataframe(machine_summary_df, use_container_width=True)


    st.markdown("#### 🧷 高スコア連番クラスタの分布と対象台番（機種名付き）")
    # 🧷 高スコア連番クラスタの分布と対象台番
    latest_df = hall_df[hall_df["日付"] == selected_date].copy()
    ranked_df = latest_df.sort_values("高設定予測スコア", ascending=False).copy()

    try:
        high_df = latest_df[latest_df["高設定予測スコア"] > 0.6].copy()
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

    # 🎨 スコア帯で色分け＆台番号に機種名を付加したランキング表示
    st.markdown("#### 🎯 スコア帯ランキング（色分け + 台番→機種名）")

    try:
        display_df = ranked_df.copy()
        display_df["前日スコア"] = display_df["前日スコア"].fillna(0.0).round(3)

        def score_color(score):
            if score > 0.8:
                return "background-color: #ff9999"  # 赤系
            elif score > 0.6:
                return "background-color: #fff599"  # 黄系
            else:
                return ""

        # 色スタイル設定
        styled = display_df.style.applymap(
            lambda v: score_color(v) if isinstance(v, float) else "", subset=["高設定予測スコア"]
        )

        # 台番号に機種名を付加（例: "101（マイジャグV）"）
        display_df["台表示"] = display_df["台番号"].astype(str) + "（" + display_df["機種名"] + "）"

        # 表示列（順番調整）
        show_cols = ["台表示", "スコア", "差枚", "G数", "高設定予測スコア", "前日スコア"]
        st.dataframe(display_df[show_cols].reset_index(drop=True).style.applymap(
            lambda v: score_color(v) if isinstance(v, float) else "", subset=["高設定予測スコア"]
        ), use_container_width=True)

    except Exception as e:
        st.warning(f"スコアランキング表示でエラーが発生しました: {e}")

    st.markdown("#### ⏪ 前日高スコアだった台の当日差枚傾向 据え置き傾向分析")

    if "前日スコア" not in latest_df.columns:
        st.warning("前日スコア列が見つかりません。")
    else:
        try:
            bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            labels = ["0.5〜0.6", "0.6〜0.7", "0.7〜0.8", "0.8〜0.9", "0.9〜1.0"]
            latest_df["前日スコア帯"] = pd.cut(latest_df["前日スコア"].fillna(0.0), bins=bins, labels=labels)

            score_trend = (
                latest_df.groupby("前日スコア帯")["差枚"]
                .agg(平均差枚="mean", 台数="count")
                .reset_index()
                .dropna()
            )

            # 勝率も追加
            plus_ratio = latest_df[latest_df["前日スコア"] > 0.6]["差枚"] > 0
            win = plus_ratio.sum()
            total = plus_ratio.count()
            st.markdown(f"✅ 差枚プラス：{win} 台（{win/total:.1%}） / マイナス：{total - win} 台（{(total - win)/total:.1%}）")

            st.dataframe(score_trend, use_container_width=True)

        except Exception as e:
            st.warning(f"前日スコアの差枚分析でエラーが発生しました: {e}")
    
    
    # 🔽 解説を追加
    st.markdown("""
    ---

    ### 📘 このランキングの活用法

    このランキングは、XGBoostモデルによって **「高設定の可能性が高い」と予測された台をスコア順に表示**したものです。

    #### 🔍 読み取りポイント
    - **機種名**：特定機種に偏りがあれば、その機種に力を入れている可能性あり
    - **台番号（末尾）**：同じ末尾が多ければ、末尾法則の可能性あり
    - **台番号（並び）**：近い番号に高スコアが並ぶ → 並び・塊の可能性
    - **差枚がマイナスでもスコアが高い**：翌日のリベンジ配置の示唆
    - **G数が少なくてもスコアが高い**：据え置きや寝かせ狙いのヒントに

    #### 🎯 使い方の例
    - 上位の台番号や機種に注目して、「傾向分析タブ」で末尾や並び傾向をさらに深掘り
    - 翌日の狙い台を仮説立てし、「仮説検証・対話」機能や自己学習ループに活用

    ---
    """)