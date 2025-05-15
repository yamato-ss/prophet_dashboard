import streamlit as st
import pandas as pd
import os
import json
from datetime import date

EVENT_MASTER_FILE = "data/event_master.json"
EVENT_SCHEDULE_FILE = "data/event_schedule.csv"

def load_event_master():
    if os.path.exists(EVENT_MASTER_FILE):
        with open(EVENT_MASTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_event_master(data):
    os.makedirs(os.path.dirname(EVENT_MASTER_FILE), exist_ok=True)
    with open(EVENT_MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_event_schedule():
    if os.path.exists(EVENT_SCHEDULE_FILE):
        return pd.read_csv(EVENT_SCHEDULE_FILE, parse_dates=["日付"])
    return pd.DataFrame(columns=["ホール", "日付", "イベント"])

def save_event_schedule(df):
    os.makedirs(os.path.dirname(EVENT_SCHEDULE_FILE), exist_ok=True)
    df.to_csv(EVENT_SCHEDULE_FILE, index=False)

def add_event_to_schedule(hall, event_date, event_name):
    df = load_event_schedule()
    # 重複削除して追加
    df = df[~((df["ホール"] == hall) & (df["日付"] == pd.Timestamp(event_date)))]
    df = pd.concat([df, pd.DataFrame([{
        "ホール": hall,
        "日付": pd.Timestamp(event_date),
        "イベント": event_name
    }])], ignore_index=True)
    save_event_schedule(df)

def event_tab():
    st.title("🎯 イベント管理")

    st.markdown("ホールを選択し、日付別にイベントを登録・編集できます。")

    # ホール選択
    hall_list = sorted(load_event_master().keys()) or ["ホールA", "ホールB"]
    selected_hall = st.selectbox("ホールを選択", hall_list)

    # イベント候補の編集
    event_dict = load_event_master()
    default_events = event_dict.get(selected_hall, [])
    st.markdown("#### イベント候補の設定")
    edited = st.text_area("カンマ区切りでイベント名を入力", ",".join(default_events))
    if st.button("イベント候補を更新"):
        event_dict[selected_hall] = [e.strip() for e in edited.split(",") if e.strip()]
        save_event_master(event_dict)
        st.success("イベント候補を更新しました")

    # 日付選択と登録
    st.markdown("#### カレンダーでイベント登録")
    selected_date = st.date_input("イベント日付", value=date.today())
    event_option = st.selectbox("登録するイベント", event_dict.get(selected_hall, []))
    if st.button("この日にイベントを登録"):
        add_event_to_schedule(selected_hall, selected_date, event_option)
        st.success(f"{selected_date} に '{event_option}' を登録しました")

    # 登録済みイベントの表示
    st.markdown("#### 登録済みイベント一覧")
    df = load_event_schedule()
    if not df.empty:
        df_view = df[df["ホール"] == selected_hall].sort_values("日付")
        st.dataframe(df_view, use_container_width=True)
    else:
        st.info("まだイベントが登録されていません。")

