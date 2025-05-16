
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import io
from PIL import Image
import os
from datetime import datetime
from utils.common import get_japanese_font, sanitize_filename

jp_font = get_japanese_font()

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

    model = Prophet(daily_seasonality=True)
    model.fit(grouped)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    today = datetime.today().strftime("%Y-%m-%d")
    safe_hall = sanitize_filename(hall_name)
    safe_machine = sanitize_filename(machine_name)
    output_dir = f"output/{safe_hall}/{safe_machine}"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grouped['ds'], grouped['y'], label="実績", linewidth=2)
    ax.plot(forecast['ds'], forecast['yhat'], label="予測", linestyle="--")
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="gray", alpha=0.3)
    ax.set_title(f"{machine_name} 差枚予測（{days}日先）", fontproperties=jp_font)
    ax.set_ylabel("平均差枚", fontproperties=jp_font)
    ax.set_xlabel("日付", fontproperties=jp_font)
    ax.legend(prop=jp_font)
    ax.grid(True)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{today}.png")
    csv_path = os.path.join(output_dir, f"{today}.csv")
    plt.savefig(png_path)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_path, index=False)
    plt.close(fig)

    buf = io.BytesIO()
    with open(png_path, "rb") as f:
        buf.write(f.read())
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def batch_forecast_for_hall(df, hall_name, days=7, force=False):
    results = []
    today = datetime.today().strftime("%Y-%m-%d")
    df['日付'] = pd.to_datetime(df['日付'])
    hall_df = df[df['ホール名'] == hall_name]
    latest_date = hall_df['日付'].max()
    filtered = hall_df[hall_df['日付'] == latest_date]

    grouped = filtered.groupby("機種名")["台番号"].nunique().reset_index(name="台数")
    machines = grouped.sort_values("台数", ascending=False)["機種名"].tolist()

    for machine_name in machines:
        target = hall_df[hall_df['機種名'] == machine_name]
        if len(target) < 10:
            continue

        grouped_data = (
            target.groupby('日付')['差枚']
            .mean()
            .reset_index()
            .rename(columns={'日付': 'ds', '差枚': 'y'})
        )

        if len(grouped_data) < 10:
            continue

        try:
            model = Prophet(daily_seasonality=True)
            model.fit(grouped_data)

            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            safe_hall = sanitize_filename(hall_name)
            safe_machine = sanitize_filename(machine_name)
            output_dir = f"output/{safe_hall}/{safe_machine}"
            os.makedirs(output_dir, exist_ok=True)
            png_path = os.path.join(output_dir, f"{today}.png")
            csv_path = os.path.join(output_dir, f"{today}.csv")

            # 予測済みチェック
            if not force and os.path.exists(png_path) and os.path.exists(csv_path):
                # results.append(f"⏭ {machine_name} - スキップ（既に予測済み）")
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(grouped_data['ds'], grouped_data['y'], label="実績", linewidth=2)
            ax.plot(forecast['ds'], forecast['yhat'], label="予測", linestyle="--")
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="gray", alpha=0.3)
            ax.set_title(f"{machine_name} 差枚予測", fontproperties=jp_font)
            ax.set_ylabel("平均差枚", fontproperties=jp_font)
            ax.set_xlabel("日付", fontproperties=jp_font)
            ax.legend(prop=jp_font)
            ax.grid(True)
            plt.tight_layout()

            png_path = os.path.join(output_dir, f"{today}.png")
            csv_path = os.path.join(output_dir, f"{today}.csv")
            plt.savefig(png_path)
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(csv_path, index=False)
            plt.close(fig)

            results.append(f"✅ {machine_name} - 予測完了")

        except Exception as e:
            results.append(f"❌ {machine_name} - エラー: {str(e)}")

    return "\n".join(results)

def batch_forecast_all(df, days=7):
    halls = df["ホール名"].dropna().unique()
    logs = []
    for hall in halls:
        logs.append(f"🏢 {hall} の予測を開始...")
        result = batch_forecast_for_hall(df, hall, days)
        logs.append(result)
    
    # バッチ全体ログ保存
    today = datetime.today().strftime("%Y-%m-%d")
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{today}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    return "\n".join(logs)