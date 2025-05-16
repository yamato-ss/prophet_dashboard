from matplotlib.font_manager import FontProperties
import re
import os
import pandas as pd

def get_japanese_font():
    return FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")

def sanitize_filename(name):
    # 禁止文字を_に置換（Windowsファイル名対策）
    return re.sub(r'[\\\\/:*?"<>|]', '_', name).strip().rstrip('.')

def load_prepared_data_for_hall(hall_name):
    """
    ホールごとの prepared_data_for_xgb_train.csv を読み込む
    """
    path = os.path.join("output", "for_xgb", hall_name, "prepared_data_for_xgb_train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")
    return pd.read_csv(path)