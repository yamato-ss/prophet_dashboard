import os
import pandas as pd
from datetime import datetime

def load_prophet_scores(output_dir="output/forecast", forecast_days=14):
    """
    各ホール・機種・日付ごとの予測結果CSVを読み込み、
    指定された対象日（ds）に対する Prophetスコア（B: 翌日, C: 3日平均）を返す
    """
    score_rows = []

    for hall_name in os.listdir(output_dir):
        hall_path = os.path.join(output_dir, hall_name)
        if not os.path.isdir(hall_path):
            continue

        for machine_name in os.listdir(hall_path):
            machine_path = os.path.join(hall_path, machine_name)
            if not os.path.isdir(machine_path):
                continue

            csv_files = [f for f in os.listdir(machine_path) if f.endswith(".csv")]
            if not csv_files:
                continue

            latest_csv = sorted(csv_files)[-1]
            csv_path = os.path.join(machine_path, latest_csv)

            try:
                df = pd.read_csv(csv_path)
                if "ds" not in df.columns:
                    raise ValueError("列 'ds' が存在しません")
                df["ds"] = pd.to_datetime(df["ds"])
                df = df.sort_values("ds")

                for i in range(min(forecast_days, len(df))):
                    row = df.iloc[i]
                    score_rows.append({
                        "ホール名": hall_name,
                        "機種名": machine_name,
                        "対象日": row["ds"].strftime("%Y-%m-%d"),
                        "Prophetスコア（yhat）": row["yhat"]
                    })

            except Exception as e:
                print(f"[ERROR] {hall_name}/{machine_name} - {e}")
                continue

    score_df = pd.DataFrame(score_rows)
    print("最終的に読み込まれたデータ")
    print(score_df.sort_values("対象日", ascending=False).head())
    return score_df
