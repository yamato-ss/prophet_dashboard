from matplotlib.font_manager import FontProperties
import re

def get_japanese_font():
    return FontProperties(fname="C:/Windows/Fonts/meiryo.ttc")

def sanitize_filename(name):
    # 禁止文字を_に置換（Windowsファイル名対策）
    return re.sub(r'[\\\\/:*?"<>|]', '_', name).strip().rstrip('.')