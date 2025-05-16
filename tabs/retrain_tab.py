# tabs/retrain_tab.py

import streamlit as st
import subprocess
import glob
import json
import os
from datetime import datetime

def show():
    st.header("🔁 自己学習ループ（Self-Training Loop）")

    # ホール一覧取得
    hall_dirs = glob.glob("output/for_xgb/*/")
    hall_names = [os.path.basename(os.path.normpath(d)) for d in hall_dirs]

    selected_hall = st.selectbox("対象ホールを選択", hall_names)
    exclude_days = st.slider("除外する直近日数", min_value=7, max_value=90, value=30, step=1)

    if st.button("🚀 自己学習を実行"):
        with st.spinner("モデル再学習中..."):
            result = subprocess.run(
                ["python", "-m", "logic.self_training_loop",
                f"--exclude-days={exclude_days}",
                f"--hall-name={selected_hall}"],
                capture_output=True,
                text=True,
                cwd=os.path.abspath(".")  # Streamlit のカレントディレクトリ明示
            )
            st.code(result.stdout)
            if result.stderr:
                st.error("⚠️ エラー発生:\n" + result.stderr)

        # ログの表示
        log_dir = f"logs/{selected_hall}/"
        logs = sorted(glob.glob(f"{log_dir}/self_train_log_*.json"), reverse=True)
        if logs:
            with open(logs[0], "r", encoding="utf-8") as f:
                log = json.load(f)

            st.success("✅ 最新の自己学習ログ")
            st.json(log)

            with open(logs[0], "rb") as f:
                st.download_button(
                    label="📥 ログファイルをダウンロード",
                    data=f,
                    file_name=os.path.basename(logs[0]),
                    mime="application/json"
                )
        else:
            st.warning("自己学習ログが見つかりませんでした。")
