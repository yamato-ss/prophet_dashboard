import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import io
from PIL import Image
import os
from datetime import datetime
import jpholiday
from utils.event_flags import apply_event_flags
from utils.common import get_japanese_font, sanitize_filename

jp_font = get_japanese_font()


def add_date_flags(df):
    df["is_weekend"] = df["ds"].dt.weekday >= 5
    df["is_zorome_day"] = df["ds"].dt.month == df["ds"].dt.day
    df["is_holiday"] = df["ds"].apply(jpholiday.is_holiday)

    # 0〜9のつく日
    for i in range(10):
        df[f"is_{i}_day"] = df["ds"].dt.day % 10 == i

    # 1〜31日の固定日（毎月）
    for d in range(1, 32):
        df[f"is_day_{d}"] = df["ds"].dt.day == d

    # 曜日（0:月曜 ～ 6:日曜）
    for i in range(7):
        df[f"is_dow_{i}"] = df["ds"].dt.weekday == i

    return df

def forecast_machine_with_prophet(df, machine_name, days=7):
    df['日付'] = pd.to_datetime(df['日付'])
    target = df[df['機種名'].str.startswith(machine_name)]
    if target.empty:
        return f"❌ 機種「{machine_name}」のデータが存在しません。"

    hall_name = target['ホール名'].iloc[0]
    grouped = (
        target.groupby('日付')['差枚']
        .mean()
        .reset_index()
        .rename(columns={'日付': 'ds', '差枚': 'y'})
    )

    if len(grouped) < 10:
        return f"⚠️ 機種「{machine_name}」の履歴が少なすぎます（{len(grouped)}件）"

    grouped["ds"] = pd.to_datetime(grouped["ds"])
    grouped = add_date_flags(grouped)
    grouped = apply_event_flags(grouped, hall_name)

    model = Prophet(daily_seasonality=True)
    regressor_cols = [col for col in grouped.columns if col.startswith(("is_", "event_"))]
    for col in regressor_cols:
        model.add_regressor(col)

    model.fit(grouped[["ds", "y"] + regressor_cols])
    future = model.make_future_dataframe(periods=days)
    future = add_date_flags(future)
    future = apply_event_flags(future, hall_name)
    forecast = model.predict(future)

    safe_hall = sanitize_filename(hall_name)
    safe_machine = sanitize_filename(machine_name)
    forecast_dir = f"output/forecast/{safe_hall}/{safe_machine}"
    eval_dir = f"output/eval/{safe_hall}/{safe_machine}"
    os.makedirs(forecast_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    last_train_day = grouped["ds"].max()
    forecast_past = forecast[forecast["ds"] <= last_train_day]
    forecast_future = forecast[forecast["ds"] > last_train_day]

    if not forecast_past.empty:
        eval_path = os.path.join(eval_dir, f"{last_train_day.strftime('%Y-%m-%d')}.csv")
        forecast_past[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(eval_path, index=False)

        # プロット（検証用）
        fig_past, ax_past = plt.subplots(figsize=(10, 4))
        ax_past.plot(grouped['ds'], grouped['y'], label="実績", linewidth=2)
        ax_past.plot(forecast_past['ds'], forecast_past['yhat'], label="予測", linestyle="--")
        ax_past.fill_between(forecast_past['ds'], forecast_past['yhat_lower'], forecast_past['yhat_upper'], color="gray", alpha=0.3)
        ax_past.set_title(f"{machine_name} 差枚予測（過去区間）", fontproperties=jp_font)
        ax_past.set_ylabel("平均差枚", fontproperties=jp_font)
        ax_past.set_xlabel("日付", fontproperties=jp_font)
        ax_past.legend(prop=jp_font)
        ax_past.grid(True)
        plt.tight_layout()
        eval_img_path = os.path.join(eval_dir, f"{last_train_day.strftime('%Y-%m-%d')}.png")
        plt.savefig(eval_img_path)
        plt.close(fig_past)

    if not forecast_future.empty:
        last_day = forecast_future["ds"].max().strftime("%Y-%m-%d")
        forecast_path = os.path.join(forecast_dir, f"{last_day}.csv")
        forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_path, index=False)

        # プロット（未来区間）
        fig_future, ax_future = plt.subplots(figsize=(10, 4))
        ax_future.plot(forecast_future['ds'], forecast_future['yhat'], label="予測", linestyle="--")
        ax_future.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], color="gray", alpha=0.3)
        ax_future.set_title(f"{machine_name} 差枚予測（未来{days}日）", fontproperties=jp_font)
        ax_future.set_ylabel("平均差枚", fontproperties=jp_font)
        ax_future.set_xlabel("日付", fontproperties=jp_font)
        ax_future.legend(prop=jp_font)
        ax_future.grid(True)
        plt.tight_layout()
        png_path = os.path.join(forecast_dir, f"{last_day}.png")
        plt.savefig(png_path)
        plt.close(fig_future)

        buf = io.BytesIO()
        with open(png_path, "rb") as f:
            buf.write(f.read())
        buf.seek(0)
        return Image.open(buf).convert("RGB"), model, forecast, grouped

    return f"⚠️ 機種「{machine_name}」の未来予測が存在しません。"

def batch_forecast_for_hall(df, hall_name, days=7, force=False):
    results = []
    forecast_last_day = None

    df["日付"] = pd.to_datetime(df["日付"])
    hall_df = df[df["ホール名"] == hall_name]
    latest_date = hall_df["日付"].max()
    filtered = hall_df[hall_df["日付"] == latest_date]
    grouped = filtered.groupby("機種名")["台番号"].nunique().reset_index(name="台数")
    machines = grouped.sort_values("台数", ascending=False)["機種名"].tolist()

    for machine_name in machines:
        target = hall_df[hall_df["機種名"] == machine_name]
        if len(target) < 10:
            continue

        grouped_data = (
            target.groupby("日付")["差枚"]
            .mean()
            .reset_index()
            .rename(columns={"日付": "ds", "差枚": "y"})
        )

        if len(grouped_data) < 10:
            continue

        try:
            grouped_data["ds"] = pd.to_datetime(grouped_data["ds"])
            grouped_data = add_date_flags(grouped_data)

            model = Prophet(daily_seasonality=True)
            regressor_cols = [col for col in grouped_data.columns if col.startswith("is_")]
            for col in regressor_cols:
                model.add_regressor(col)

            model.fit(grouped_data[["ds", "y"] + regressor_cols])
            future = model.make_future_dataframe(periods=days)
            future = add_date_flags(future)
            forecast = model.predict(future)
            forecast_last_day = forecast["ds"].max()

            safe_hall = sanitize_filename(hall_name)
            safe_machine = sanitize_filename(machine_name)
            forecast_dir = f"output/forecast/{safe_hall}/{safe_machine}"
            eval_dir = f"output/eval/{safe_hall}/{safe_machine}"
            os.makedirs(forecast_dir, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)

            last_train_day = grouped_data["ds"].max()
            forecast_past = forecast[forecast["ds"] <= last_train_day]
            forecast_future = forecast[forecast["ds"] > last_train_day]

            # eval出力
            if not forecast_past.empty:
                eval_csv_path = os.path.join(eval_dir, f"{last_train_day.strftime('%Y-%m-%d')}.csv")
                forecast_past[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(eval_csv_path, index=False)

                fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
                ax_eval.plot(grouped_data["ds"], grouped_data["y"], label="実績", linewidth=2)
                ax_eval.plot(forecast_past["ds"], forecast_past["yhat"], label="予測", linestyle="--")
                ax_eval.fill_between(forecast_past["ds"], forecast_past["yhat_lower"], forecast_past["yhat_upper"], color="gray", alpha=0.3)
                ax_eval.set_title(f"{machine_name} 差枚予測（過去区間）", fontproperties=jp_font)
                ax_eval.set_ylabel("平均差枚", fontproperties=jp_font)
                ax_eval.set_xlabel("日付", fontproperties=jp_font)
                ax_eval.legend(prop=jp_font)
                ax_eval.grid(True)
                plt.tight_layout()
                eval_img_path = os.path.join(eval_dir, f"{last_train_day.strftime('%Y-%m-%d')}.png")
                plt.savefig(eval_img_path)
                plt.close(fig_eval)

            # forecast出力
            if not forecast_future.empty:
                last_day = forecast_future["ds"].max().strftime("%Y-%m-%d")
                forecast_csv_path = os.path.join(forecast_dir, f"{last_day}.csv")
                forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_csv_path, index=False)

                fig_forecast, ax_forecast = plt.subplots(figsize=(10, 4))
                ax_forecast.plot(forecast_future["ds"], forecast_future["yhat"], label="予測", linestyle="--")
                ax_forecast.fill_between(forecast_future["ds"], forecast_future["yhat_lower"], forecast_future["yhat_upper"], color="gray", alpha=0.3)
                ax_forecast.set_title(f"{machine_name} 差枚予測（未来{days}日）", fontproperties=jp_font)
                ax_forecast.set_ylabel("平均差枚", fontproperties=jp_font)
                ax_forecast.set_xlabel("日付", fontproperties=jp_font)
                ax_forecast.legend(prop=jp_font)
                ax_forecast.grid(True)
                plt.tight_layout()
                forecast_img_path = os.path.join(forecast_dir, f"{last_day}.png")
                plt.savefig(forecast_img_path)
                plt.close(fig_forecast)

            results.append(f"✅ {machine_name} - 予測完了")

        except Exception as e:
            results.append(f"❌ {machine_name} - エラー: {str(e)}")

    return "\n".join(results), forecast_last_day.strftime("%Y-%m-%d") if forecast_last_day else None

def batch_forecast_all(df, days=7):
    halls = df["ホール名"].dropna().unique()
    logs = []
    last_day = None

    for hall in halls:
        logs.append(f"🏢 {hall} の予測を開始...")
        result, forecast_day = batch_forecast_for_hall(df, hall, days)
        logs.append(result)
        last_day = forecast_day

    if last_day:
        log_dir = "output/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{last_day}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))

    return "\n".join(logs)