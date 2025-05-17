import streamlit as st

# 各タブの読み込み
from tabs.preprocess_tab import show as preprocess_tab
from tabs.train_tab import show as train_tab
from tabs.retrain_tab import show as retrain_tab
from tabs.prophet_tab import show as prophet_tab
from tabs.xgb_score_tab import score_tab as xgb_score_tab
from tabs.xgb_prophet_tab import score_tab as xgb_prophet_tab
from tabs.event_tab import event_tab  # 追加

# アプリ全体の説明を表示
st.markdown("""
# 🎰 パチンコ狙い台分析ダッシュボード

このダッシュボードでは、以下のカテゴリに分かれた機能でデータ分析・狙い台予測を行います：

## 🧪 データ生成・操作
- CSV結合・前処理（`前処理`）
- 機種ごとの差枚予測（`Prophet予測`）
- 台ごとの高設定スコア学習（`モデル学習`）
- 自己学習ループで精度向上（`自己学習`）
- ホールイベントの登録（`イベント登録`）

## 🎯 スコア分析・狙い台表示
- XGBoost単体によるスコアランキング（`XGBoostスコア`）
- Prophet × XGBoostのAND条件で狙い台抽出（`ANDスコア統合`）

---

## 🔁 操作の流れ
1. **前処理**：CSVファイルを統合し、学習用データを生成します
2. **モデル学習**：XGBoostモデルをホール単位で学習します
3. **Prophet予測**：機種ごとに未来差枚を予測（CSVとして保存）
4. **自己学習（任意）**：予測誤差の改善を目的に再学習します
5. **XGBoostスコア確認**：高設定スコア上位の台を確認します
6. **ANDスコア統合**：Prophet × XGBoostで狙い台を抽出します

---

📌 サイドバーからカテゴリと機能を選んで操作してください。
""")

# カテゴリ別でタブを整理
st.sidebar.title("📁 タブカテゴリ")
category = st.sidebar.radio("カテゴリを選択", ["🧪 データ生成・操作", "🎯 スコア分析・狙い台表示"])

if category == "🧪 データ生成・操作":
    tab = st.sidebar.radio("機能を選択", ["前処理", "Prophet予測", "モデル学習", "自己学習", "イベント登録"])
    if tab == "前処理":
        preprocess_tab()
    elif tab == "Prophet予測":
        prophet_tab()
    elif tab == "モデル学習":
        train_tab()
    elif tab == "自己学習":
        retrain_tab()
    elif tab == "イベント登録":
        event_tab()

elif category == "🎯 スコア分析・狙い台表示":
    tab = st.sidebar.radio("機能を選択", ["XGBoostスコア", "ANDスコア統合"])
    if tab == "XGBoostスコア":
        xgb_score_tab()
    elif tab == "ANDスコア統合":
        xgb_prophet_tab()
