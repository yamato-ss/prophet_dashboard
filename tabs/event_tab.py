import streamlit as st
from streamlit_calendar import calendar
import pandas as pd
import os
from datetime import date, datetime, timedelta

EVENT_BASE_DIR = "data/events"
FOR_XGB_DIR = "output/for_xgb"

def load_hall_list():
    return sorted([
        name for name in os.listdir(FOR_XGB_DIR)
        if os.path.isdir(os.path.join(FOR_XGB_DIR, name))
    ]) if os.path.exists(FOR_XGB_DIR) else []

def get_event_dir(hall):
    return os.path.join(EVENT_BASE_DIR, hall)

def load_event_master(hall):
    path = os.path.join(get_event_dir(hall), "event.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["イベント名", "tag"])

def load_event_days(hall):
    path = os.path.join(get_event_dir(hall), "event_day.csv")
    return pd.read_csv(path, parse_dates=["日付"]) if os.path.exists(path) else pd.DataFrame(columns=["日付", "イベント名"])

def save_event_days(hall, df):
    path = os.path.join(get_event_dir(hall), "event_day.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def convert_to_calendar_events(df, event_master_df):
    tag_to_name = dict(zip(event_master_df["tag"], event_master_df["イベント名"]))
    return [
        {
            "id": f"{row['日付'].date()}__{row['イベント名']}",
            "title": tag_to_name.get(row["イベント名"], row["イベント名"]),
            "start": row["日付"].strftime("%Y-%m-%d"),
        }
        for _, row in df.iterrows()
    ]

# ✅ どちらの形式でも日付を JST で返す
def parse_date_or_datetime(iso_str):
    try:
        return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%S.000Z") + timedelta(hours=9)
    except ValueError:
        return datetime.strptime(iso_str, "%Y-%m-%d")  # JSTとして扱う（既にJST）

def event_tab():
    st.title("📆 カレンダーイベント管理")

    hall_list = load_hall_list()
    if not hall_list:
        st.warning("ホールデータが見つかりません")
        return

    selected_hall = st.selectbox("ホールを選択", hall_list)
    event_master = load_event_master(selected_hall)
    event_days_df = load_event_days(selected_hall)

    # セッション初期化
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()
    if "selected_event_tag" not in st.session_state:
        st.session_state["selected_event_tag"] = ""

    st.subheader("🗓️ カレンダー表示")
    calendar_events = convert_to_calendar_events(event_days_df, event_master)
    cal_output = calendar(events=calendar_events, options={"initialView": "dayGridMonth", "selectable": True, "height": 600})

    # ✅ 日付クリック（UTC → JST）+ 選択状態解除
    if cal_output.get("dateClick"):
        utc_str = cal_output["dateClick"]["date"]
        selected_date = parse_date_or_datetime(utc_str).date()
        st.session_state["selected_date"] = selected_date

        # 選択した日付に登録がなければ、イベント選択状態を解除
        if not any((event_days_df["日付"].dt.date == selected_date)):
            st.session_state["selected_event_tag"] = ""

    # ✅ イベントクリック
    event_info = cal_output.get("eventClick", {}).get("event", {})
    event_id = event_info.get("id")
    start_utc = event_info.get("start")

    if event_id and "__" in event_id and start_utc:
        d_str, tag = event_id.split("__")
        st.session_state["selected_date"] = parse_date_or_datetime(start_utc).date()
        st.session_state["selected_event_tag"] = tag

    selected_date = st.session_state["selected_date"]
    selected_event_tag = st.session_state["selected_event_tag"]

    # 🛠 編集／削除UI（選択中のみ表示）
    if not event_master.empty and selected_event_tag:
        st.markdown("### 🛠 選択中のイベントを操作")
        display_name = event_master.set_index("tag").get("イベント名", {}).get(selected_event_tag, selected_event_tag)

        if st.button(f"🗑️ {selected_date} の「{display_name}」を削除する"):
            df = event_days_df[
                ~((event_days_df["日付"].dt.date == selected_date) & (event_days_df["イベント名"] == selected_event_tag))
            ]
            save_event_days(selected_hall, df)
            st.success(f"{selected_date} の「{display_name}」を削除しました")
            st.session_state["selected_event_tag"] = ""
            st.rerun()

    # ➕ イベント追加UI
    st.subheader("➕ イベントを追加")
    st.markdown(f"📅 選択中の日付： `{selected_date}`")

    if not event_master.empty:
        options = event_master.to_dict("records")
        display_list = [e["イベント名"] for e in options]

        selected_index = st.selectbox(
            "登録するイベントを選択",
            range(len(options)),
            format_func=lambda i: display_list[i]
        )
        selected_tag = options[selected_index]["tag"]
        selected_name = options[selected_index]["イベント名"]

        if st.button("登録 / 上書き保存"):
            new_row = pd.DataFrame([{
                "日付": pd.to_datetime(selected_date),
                "イベント名": selected_tag
            }])
            new_df = pd.concat([event_days_df, new_row], ignore_index=True).drop_duplicates()
            save_event_days(selected_hall, new_df)
            st.success(f"{selected_date} に「{selected_name}」を登録しました")
            st.rerun()

    st.subheader("📋 現在の登録イベント一覧")
    st.dataframe(event_days_df.sort_values("日付"))