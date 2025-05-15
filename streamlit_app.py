import streamlit as st
from tabs.prophet_tab import prophet_tab
from tabs.xgboost_tab import show as xgboost_tab
from tabs.trend_tab import show as trend_tab
from tabs.preprocess_tab import show as preprocess_tab
from tabs.train_tab import show as train_tab
from tabs.score_tab import show as score_tab

st.set_page_config(layout="wide")
st.title("🎰 パチスロ狙い台分析ダッシュボード")

main_tabs = st.tabs(["📊 分析・閲覧", "⚙️ データ生成・操作"])

with main_tabs[0]:
    view_option = st.radio("表示内容を選択", [
        "🔮 機種別・差枚の未来予測（Prophet）",
        "🎯 高設定スコア予測ランキング（XGBoost）",
        "🔍 傾向分析（末尾・並び）"
    ])

    if view_option == "🔮 機種別・差枚の未来予測（Prophet）":
        prophet_tab()
    elif view_option == "🎯 高設定スコア予測ランキング（XGBoost）":
        xgboost_tab()
    elif view_option == "🔍 傾向分析（末尾・並び）":
        trend_tab()

with main_tabs[1]:
    # 🔧 操作の流れを表示（radioの前に）
    st.markdown("""
    ### 🧭 操作の流れ（推奨ステップ）

    1. **⚙️ データ統合・特徴量生成**  
       → 複数ホールのCSVを統合し、`prepared_data.csv` を出力します。

    2. **✎ XGBoostモデル学習**  
       → `prepared_data.csv` を使ってモデルを学習し、`models/xxx.json` を出力します。

    3. **⚡ 高設定スコア出力**  
       → モデルを選んで `predicted_with_score.csv` を出力。  
       → ランキング表示タブで確認できます。
    """, unsafe_allow_html=True)

    op_option = st.radio("操作内容を選択", [
        "⚙️ データ統合・特徴量生成",
        "✎XGBoostモデル学習",
        "⚡ 高設定スコア出力"
    ])

    if op_option == "⚙️ データ統合・特徴量生成":
        preprocess_tab()
    elif op_option == "✎XGBoostモデル学習":
        train_tab()
    elif op_option == "⚡ 高設定スコア出力":
        score_tab()
