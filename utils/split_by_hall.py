import os
import pandas as pd

def split_prepared_data_by_hall(input_path="output/prepared_data.csv", output_root="output/for_xgb"):
    df = pd.read_csv(input_path)

    if "ホール名" not in df.columns:
        raise ValueError("CSVに 'ホール名' カラムが存在しません。")

    hall_list = df["ホール名"].unique()

    for hall_name in hall_list:
        df_hall = df[df["ホール名"] == hall_name]
        hall_dir = os.path.join(output_root, hall_name)
        os.makedirs(hall_dir, exist_ok=True)
        out_path = os.path.join(hall_dir, "prepared_data_for_xgb_train.csv")
        df_hall.to_csv(out_path, index=False)

    return hall_list
