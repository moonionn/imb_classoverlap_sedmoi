from imblearn.datasets import fetch_datasets
import pandas as pd
import numpy as np
import os


def save_datasets_as_csv():
    """載入所有需要的datasets並保存為CSV"""

    # 創建CSV資料夾
    csv_folder = "./csv_datasets"
    os.makedirs(csv_folder, exist_ok=True)

    # Dataset ID對應名稱
    dataset_names = {
        1: 'ecoli',
        2: 'optical_digits',
        5: 'abalone',
        6: 'sick_euthyroid',
        7: 'spectrometer',
        8: 'car_eval_34',
        10: 'us_crime',
        11: 'yeast_ml8',
        12: 'scene',
        13: 'libras_move',
        14: 'thyroid_sick',
        17: 'solar_flare_m0',
        18: 'oil',
        19: 'car_eval_4',
        20: 'wine_quality',
        22: 'yeast_me2',
        24: 'ozone_level',
        27: 'abalone_19'
    }

    # 要排除的dataset IDs
    exclude_ids = [3, 4, 9, 15, 16, 21, 23, 25, 26]
    include_ids = [id for id in range(1, 28) if id not in exclude_ids]

    print("開始載入datasets...")

    # 載入所有datasets
    print("正在下載所有datasets（這可能需要一些時間）...")
    all_datasets = fetch_datasets(verbose=True)

    print(f"成功載入 {len(all_datasets)} 個datasets")
    print("可用的datasets:", list(all_datasets.keys()))

    # 處理每個我們需要的dataset
    dataset_info = []

    for dataset_id in include_ids:
        if dataset_id in dataset_names:
            dataset_name = dataset_names[dataset_id]

            if dataset_name in all_datasets:
                print(f"\n處理 Dataset ID {dataset_id}: {dataset_name}")

                # 獲取dataset
                dataset = all_datasets[dataset_name]
                X = dataset.data
                y = dataset.target

                # 創建DataFrame
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_names)
                df['target'] = y

                # 保存為CSV
                csv_filename = f"{dataset_name}.csv"
                csv_path = os.path.join(csv_folder, csv_filename)
                df.to_csv(csv_path, index=False)

                # 計算統計信息
                class_counts = df['target'].value_counts().sort_index()
                ir_ratio = class_counts.max() / class_counts.min()

                dataset_info.append({
                    'ID': dataset_id,
                    'Name': dataset_name,
                    'Samples': len(df),
                    'Features': X.shape[1],
                    'Classes': len(class_counts),
                    'IR_Ratio': f"{ir_ratio:.1f}:1",
                    'Class_Distribution': dict(class_counts)
                })

                print(f"  已保存: {csv_filename}")
                print(f"  形狀: {df.shape}")
                print(f"  不平衡比率: {ir_ratio:.1f}:1")
                print(f"  類別分布: {dict(class_counts)}")

            else:
                print(f"警告: Dataset '{dataset_name}' 不在載入的datasets中")

    # 創建摘要文件
    if dataset_info:
        summary_df = pd.DataFrame(dataset_info)
        summary_path = os.path.join(csv_folder, "datasets_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n轉換完成！")
        print(f"CSV檔案保存在: {os.path.abspath(csv_folder)}")
        print(f"摘要資訊已保存為: datasets_summary.csv")

        print("\n數據集摘要:")
        print(summary_df[['ID', 'Name', 'Samples', 'Features', 'IR_Ratio']])

        return summary_df
    else:
        print("沒有成功處理任何數據集")
        return None


def load_specific_datasets():
    """只載入我們需要的特定datasets (更快的版本)"""

    # 創建CSV資料夾
    csv_folder = "./csv_datasets"
    os.makedirs(csv_folder, exist_ok=True)

    # 只載入我們需要的datasets
    needed_datasets = [
        'ecoli', 'optical_digits', 'abalone', 'sick_euthyroid',
        'spectrometer', 'car_eval_34', 'us_crime', 'yeast_ml8',
        'scene', 'libras_move', 'thyroid_sick', 'solar_flare_m0',
        'oil', 'car_eval_4', 'wine_quality', 'yeast_me2',
        'ozone_level', 'abalone_19'
    ]

    # ID對應
    id_mapping = {
        'ecoli': 1, 'optical_digits': 2, 'abalone': 5, 'sick_euthyroid': 6,
        'spectrometer': 7, 'car_eval_34': 8, 'us_crime': 10, 'yeast_ml8': 11,
        'scene': 12, 'libras_move': 13, 'thyroid_sick': 14, 'solar_flare_m0': 17,
        'oil': 18, 'car_eval_4': 19, 'wine_quality': 20, 'yeast_me2': 22,
        'ozone_level': 24, 'abalone_19': 27
    }

    print("開始載入指定的datasets...")

    dataset_info = []

    for dataset_name in needed_datasets:
        try:
            print(f"載入 {dataset_name}...")

            # 載入單個dataset
            dataset_dict = fetch_datasets(filter_data=(dataset_name,))
            dataset = dataset_dict[dataset_name]

            X = dataset.data
            y = dataset.target

            # 創建DataFrame
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y

            # 保存為CSV
            csv_filename = f"{dataset_name}.csv"
            csv_path = os.path.join(csv_folder, csv_filename)
            df.to_csv(csv_path, index=False)

            # 計算統計信息
            class_counts = df['target'].value_counts().sort_index()
            ir_ratio = class_counts.max() / class_counts.min()

            dataset_info.append({
                'ID': id_mapping[dataset_name],
                'Name': dataset_name,
                'Samples': len(df),
                'Features': X.shape[1],
                'Classes': len(class_counts),
                'IR_Ratio': f"{ir_ratio:.1f}:1",
                'Class_Distribution': dict(class_counts)
            })

            print(f"  已保存: {csv_filename}")
            print(f"  形狀: {df.shape}")
            print(f"  不平衡比率: {ir_ratio:.1f}:1")

        except Exception as e:
            print(f"載入 {dataset_name} 時發生錯誤: {e}")

    # 創建摘要文件
    if dataset_info:
        summary_df = pd.DataFrame(dataset_info)
        summary_df = summary_df.sort_values('ID').reset_index(drop=True)
        summary_path = os.path.join(csv_folder, "datasets_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n轉換完成！")
        print(f"CSV檔案保存在: {os.path.abspath(csv_folder)}")
        print(f"摘要資訊已保存為: datasets_summary.csv")

        print("\n數據集摘要:")
        print(summary_df[['ID', 'Name', 'Samples', 'Features', 'IR_Ratio']])

        return summary_df
    else:
        print("沒有成功處理任何數據集")
        return None


if __name__ == "__main__":
    # 使用更快的版本（推薦）
    summary = load_specific_datasets()

    # 如果想載入所有datasets再篩選，使用這個：
    # summary = save_datasets_as_csv()