# llm_utils.py

import requests

def generate_event_summary(prompt, api_url="http://127.0.0.1:5000/v1/chat/completions"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-3.5-turbo",  # 実際のモデル名に応じて変更可
        "messages": [
            {"role": "system", "content": "あなたはパチンコ店のイベント傾向を分析する専門家です。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        res = requests.post(api_url, json=payload, headers=headers)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ エラー: {e}"
    
def build_event_summary_prompt(hall_name, event_df, df_all):
    prompt = f"ホール「{hall_name}」におけるイベントと実際の高設定率の記録です。\n"
    prompt += "この中で高設定投入傾向が強かったタイプや傾向を日本語で要約してください。\n\n"
    prompt += "■イベント履歴：\n"

    for _, row in event_df.iterrows():
        date = row["日付"].strftime("%Y-%m-%d")
        target = str(row["対象"])
        type_ = row["タイプ"]
        event_day_df = df_all[(df_all["日付"] == row["日付"]) & (df_all["ホール名"] == hall_name)]
        if type_ == "末尾":
            event_day_df = event_day_df[event_day_df["台番号"].astype(str).str.endswith(target)]
        elif type_ == "機種":
            event_day_df = event_day_df[event_day_df["機種名"].str.contains(target, na=False)]

        if len(event_day_df) > 0:
            rate = event_day_df["高設定"].mean() * 100
            prompt += f"- {date}：{row['イベント内容']} → 高設定率 {rate:.1f}%（{len(event_day_df)}台）\n"
        else:
            prompt += f"- {date}：{row['イベント内容']}（該当台なし）\n"

    prompt += "\n平均差枚・台数も考慮し、端的にまとめてください。"
    return prompt