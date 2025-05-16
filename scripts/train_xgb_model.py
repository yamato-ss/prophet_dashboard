# CLI実行用モデル学習実行スクリプト
# 実行例
# 学習のみ（保存しない）
# python scripts/train_xgb_model.py

# モデル保存付き
# python scripts/train_xgb_model.py --save



import argparse
from logic.train_utils import load_prepared_data, train_model, save_model
from pprint import pprint

def main(input_file="output/prepared_for_xgb.csv", save=False):
    print("🔄 データ読み込み中...")
    df = load_prepared_data(input_file)

    print("🧠 モデル学習中...")
    model, auc, report = train_model(df)

    print(f"\n🎯 ROC AUCスコア: {auc:.4f}")
    print("\n📋 評価レポート:")
    pprint(report)

    if save:
        path = save_model(model)
        print(f"\n💾 モデルを保存しました: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoostモデルの学習と保存")
    parser.add_argument("--input", type=str, default="output/prepared_for_xgb.csv", help="学習用CSVファイルパス")
    parser.add_argument("--save", action="store_true", help="学習後にモデルを保存する")

    args = parser.parse_args()
    main(args.input, args.save)
