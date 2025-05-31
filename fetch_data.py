from imblearn.datasets import fetch_datasets
import pandas as pd
import numpy as np
import os

""" Collection of imbalanced datasets.
ID    Name           Repository & Target           Ratio  #S       #F
 1     ecoli          UCI, target: imU              8.6:1  336      7
 2     optical_digits UCI, target: 8                9.1:1  5,620    64
 3     satimage       UCI, target: 4                9.3:1  6,435    36
 4     pen_digits     UCI, target: 5                9.4:1  10,992   16
 5     abalone        UCI, target: 7                9.7:1  4,177    10
 6     sick_euthyroid UCI, target: sick euthyroid   9.8:1  3,163    42
 7     spectrometer   UCI, target: >=44             11:1   531      93
 8     car_eval_34    UCI, target: good, v good     12:1   1,728    21
 9     isolet         UCI, target: A, B             12:1   7,797    617
 10    us_crime       UCI, target: >0.65            12:1   1,994    100
 11    yeast_ml8      LIBSVM, target: 8             13:1   2,417    103
 12    scene          LIBSVM, target: >one label    13:1   2,407    294
 13    libras_move    UCI, target: 1                14:1   360      90
 14    thyroid_sick   UCI, target: sick             15:1   3,772    52
 15    coil_2000      KDD, CoIL, target: minority   16:1   9,822    85
 16    arrhythmia     UCI, target: 06               17:1   452      278
 17    solar_flare_m0 UCI, target: M->0             19:1   1,389    32
 18    oil            UCI, target: minority         22:1   937      49
 19    car_eval_4     UCI, target: vgood            26:1   1,728    21
 20    wine_quality   UCI, wine, target: <=4        26:1   4,898    11
 21    letter_img     UCI, target: Z                26:1   20,000   16
 22    yeast_me2      UCI, target: ME2              28:1   1,484    8
 23    webpage        LIBSVM, w7a, target: minority 33:1   34,780   300
 24    ozone_level    UCI, ozone, data              34:1   2,536    72
 25    mammography    UCI, target: minority         42:1   11,183   6
 26    protein_homo   KDD CUP 2004, minority        111:1  145,751  74
 27    abalone_19     UCI, target: 19               130:1  4,177    10
"""

def download_all_datasets():
    """下載所有可用的datasets並保存為CSV"""
    
    # 創建CSV資料夾
    csv_folder = "./datasets"
    os.makedirs(csv_folder, exist_ok=True)
    
    print("開始載入所有可用的datasets...")
    
    # 載入所有datasets
    print("正在下載所有datasets（這可能需要很長時間）...")
    all_datasets = fetch_datasets(verbose=True)
    
    print(f"成功載入 {len(all_datasets)} 個datasets")
    print("可用的datasets:", list(all_datasets.keys()))
    
    # 處理所有可用的dataset
    dataset_info = []
    
    for i, dataset_name in enumerate(all_datasets.keys(), 1):
        try:
            print(f"\n處理 Dataset {i}/{len(all_datasets)}: {dataset_name}")
            
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
                'ID': i,
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
            
        except Exception as e:
            print(f"處理 {dataset_name} 時發生錯誤: {e}")
            continue
    
    # 創建摘要文件
    if dataset_info:
        summary_df = pd.DataFrame(dataset_info)
        summary_path = os.path.join(csv_folder, "all_datasets_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n轉換完成！")
        print(f"CSV檔案保存在: {os.path.abspath(csv_folder)}")
        print(f"摘要資訊已保存為: all_datasets_summary.csv")
        print(f"總共處理了 {len(dataset_info)} 個資料集")
        
        print("\n數據集摘要:")
        print(summary_df[['ID', 'Name', 'Samples', 'Features', 'IR_Ratio']])
        
        return summary_df
    else:
        print("沒有成功處理任何數據集")
        return None


if __name__ == "__main__":
    summary = download_all_datasets()
