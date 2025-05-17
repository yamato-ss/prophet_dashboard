import pandas as pd
import os

def apply_event_flags(df: pd.DataFrame, hall_name: str, base_dir: str = "data/events") -> pd.DataFrame:
    """
    `data/events/{ホール名}/event_day.csv` をもとに、
    df["日付"] に event_〇〇 のフラグ列を追加する。
    """
    path = os.path.join(base_dir, hall_name, "event_day.csv")
    if not os.path.exists(path):
        return df  # ファイルが存在しない場合は何もせず返す

    event_df = pd.read_csv(path, parse_dates=["日付"])
    if event_df.empty:
        return df

    # 正規化（日付のみ比較）
    df["日付"] = pd.to_datetime(df["日付"])
    df_dates = df["日付"].dt.normalize()

    for event_type in event_df["イベント名"].unique():
        event_dates = event_df[event_df["イベント名"] == event_type]["日付"].dt.normalize()
        df[f"event_{event_type}"] = df_dates.isin(event_dates).astype(int)

    return df
