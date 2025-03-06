import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 全局配置
DATA_PATH = {
    'raw': './data/raw',
    'clean': './data/clean',
    'processed': './data/processed'
}

# 胰岛素数据处理配置
TIME_SEGMENTS = [
    (0, 4),     # 00:00-04:00
    (4, 9),     # 04:00-09:00
    (9, 12),    # 09:00-12:00
    (12, 17),   # 12:00-17:00
    (17, 21),   # 17:00-21:00
    (21, 24)    # 21:00-24:00
]

MEAL_HOURS = {
    'breakfast': 8,
    'lunch': 12,
    'dinner': 18
}

# 患者元数据
PATIENT_METADATA = {
    "LT2310KDPQ": {"age": 45, "gender": "M", "weight": 117, "insulin": "", "calories": 1800, "drug": ""},
    "LT2401MV1C": {"age": 56, "gender": "M", "weight": None, "insulin": "", "calories": 1800, "drug": ""},
    "LT2401MVLD": {"age": 72, "gender": "M", "weight": 99, "insulin": "", "calories": 1800, "drug": ""},
    "LT2401MVLT": {"age": 66, "gender": "F", "weight": 51.5, "insulin": "", "calories": 1400, "drug": ""},
    "LT2401P3JN": {"age": 43, "gender": "M", "weight": 83, "insulin": "", "calories": 1600, "drug": ""},
    "LT2401P3JP": {"age": 62, "gender": "M", "weight": 62.5, "insulin": "", "calories": 1800, "drug": ""},
    "LT2401P49L": {"age": 62, "gender": "M", "weight": 51.5, "insulin": "", "calories": 1800, "drug": ""},
    "LT2401PQ5K": {"age": 61, "gender": "F", "weight": 45, "insulin": "", "calories": 1600, "drug": ""},
    "LT2401PQLY": {"age": 71, "gender": "F", "weight": 64.5, "insulin": "", "calories": 1400, "drug": ""},
    "LT231109MQ": {"age": 63, "gender": "F", "weight": 77, "insulin": "", "calories": 1400, "drug": ""}
}

def get_patient_metadata(file_path):
    """从文件中获取患者元数据"""
    metadata = {}

def onehot(df):
    """对数据进行独热编码处理
    
    Args:
        df: 输入的DataFrame，包含需要进行独热编码的列
    
    Returns:
        处理后的DataFrame，包含独热编码后的特征
    """
    try:
        processed_df = df.copy()
        
        if 'gender' in processed_df.columns:
            gender_dummies = pd.get_dummies(
                processed_df['gender'], 
                prefix='gender',
                drop_first=True
            )
            processed_df = pd.concat([processed_df.drop('gender', axis=1), gender_dummies], axis=1)
        
        return processed_df
        
    except Exception as e:
        return df

def process_original_data(raw_path):
    """处理原始数据并合并
    
    Args:
        raw_path: 原始数据文件夹路径
    
    Returns:
        处理后的合并数据DataFrame，时间戳规整到整点
    """
    def extract_patient_id(filename):
        """从文件名中提取病人ID"""
        parts = filename.split('-')
        if len(parts) >= 3:
            return parts[2]
        return None
    
    all_data = []
    
    # 遍历原始数据文件夹
    for file_name in os.listdir(raw_path):
        if not file_name.endswith('.xlsx'):
            continue
            
        patient_id = extract_patient_id(file_name)
        if not patient_id:
            print(f"警告：无法从文件名 {file_name} 提取病人ID")
            continue
            
        try:
            file_path = os.path.join(raw_path, file_name)
            df = pd.read_excel(file_path)
            
            # 检查必要的列是否存在
            required_cols = ['采集时间', '血糖值(mmol/L)']
            if not all(col in df.columns for col in required_cols):
                print(f"错误：文件 {file_name} 缺少必要的列")
                continue
                
            # 检查数据是否需要翻转
            if df['采集时间'].iloc[0] > df['采集时间'].iloc[-1]:
                df = df.iloc[::-1].reset_index(drop=True)
            
            # 基础数据处理
            df['item_id'] = patient_id
            df['original_time'] = pd.to_datetime(df['采集时间'])
            df['血糖值(mmol/L)'] = pd.to_numeric(df['血糖值(mmol/L)'], errors='coerce')
            df = df.dropna(subset=['血糖值(mmol/L)'])
            
            # 时间戳处理
            df['timestamp'] = df['original_time'].dt.round('H')
            df_sorted = df.sort_values(['timestamp', 'original_time'])
            df_deduplicated = df_sorted.drop_duplicates(
                subset=['item_id', 'timestamp'], 
                keep='first'
            )
            df_processed = df_deduplicated.rename(columns={'血糖值(mmol/L)': 'target'})
            df_processed = df_processed[['item_id', 'timestamp', 'target']]
            
            all_data.append(df_processed)
            print(f"原始记录数：{len(df)}，处理后记录数：{len(df_processed)}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错：{str(e)}")
            continue
    
    if not all_data:
        print("错误：没有成功处理任何文件！")
        return None
        
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    
    # 保存合并后的数据
    output_file = os.path.join(DATA_PATH['processed'], 'merged_raw.csv')
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n原始数据处理完成！")
    print(f"总记录数：{len(merged_df)}")
    print(f"病人数量：{merged_df['item_id'].nunique()}")
    print(f"时间范围：{merged_df['timestamp'].min()} 至 {merged_df['timestamp'].max()}")
    print(f"数据已保存至：{output_file}")
    
    return merged_df

def process_enriched_data(input_file):
    """处理原始数据，添加患者元数据
    
    Args:
        input_file: merged_raw.csv的路径
    
    Returns:
        添加元数据后的DataFrame
    """
    try:
        # 读取原始合并数据
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 添加元数据
        enriched_data = []
        for item_id in df['item_id'].unique():
            # 获取该患者的数据
            patient_df = df[df['item_id'] == item_id].copy()
            
            # 获取患者元数据
            meta = PATIENT_METADATA.get(item_id, {})
            
            # 添加元数据字段
            patient_df['age'] = meta.get('age', None)
            patient_df['gender'] = meta.get('gender', None)
            # patient_df['type'] = meta.get('type', None)
            patient_df['weight'] = meta.get('weight', None)
            patient_df['insulin'] = meta.get('insulin', None)
            patient_df['calories'] = meta.get('calories', None)
            patient_df['drug'] = meta.get('drug', '')
            
            enriched_data.append(patient_df)
            print(f"处理患者 {item_id} 数据：{len(patient_df)} 条记录")
        
        # 合并所有数据
        enriched_df = pd.concat(enriched_data, ignore_index=True)
        
        # 数据类型转换
        enriched_df['age'] = enriched_df['age'].astype('Int64')
        # enriched_df['type'] = enriched_df['type'].astype('Int64')
        enriched_df['weight'] = pd.to_numeric(enriched_df['weight'], errors='coerce')
        enriched_df['calories'] = pd.to_numeric(enriched_df['calories'], errors='coerce')
        
        # 性别独热编码处理
        enriched_df = onehot(enriched_df)
        
        # 按时间排序
        enriched_df = enriched_df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
        
        # 保存结果
        output_file = os.path.join(DATA_PATH['processed'], 'merged_enriched.csv')
        enriched_df.to_csv(output_file, index=False)
        
        print(f"\n数据处理完成！")
        print(f"总记录数：{len(enriched_df)}")
        print(f"患者数量：{enriched_df['item_id'].nunique()}")
        print(f"时间范围：{enriched_df['timestamp'].min()} 至 {enriched_df['timestamp'].max()}")
        print(f"数据已保存至：{output_file}")
        
        return enriched_df
        
    except Exception as e:
        print(f"处理数据时出错：{str(e)}")
        return None

def process_insulin_data(input_file):
    """处理胰岛素数据"""
    df = pd.read_csv(input_file)
    expanded_data = []
    
    for _, row in df.iterrows():
        coefficients = list(map(float, row['basal_coefficient'].split('-')))
        date = datetime.strptime(row['timestamp'], '%Y/%m/%d')
        daily_total = row['basal_total']
        
        for hour in range(24):
            basal_rate = next(
                (coefficients[i] 
                 for i, (start, end) in enumerate(TIME_SEGMENTS)
                 if start <= hour < end),
                None
            )
            
            meal_bolus = 0
            if hour == MEAL_HOURS['breakfast']:
                meal_bolus = row['bolus_breakfast']
            elif hour == MEAL_HOURS['lunch']:
                meal_bolus = row['bolus_lunch']
            elif hour == MEAL_HOURS['dinner']:
                meal_bolus = row['bolus_dinner']
            
            basal_hourly = daily_total * basal_rate
            
            expanded_data.append({
                'item_id': row['item_id'],
                'timestamp': (date + timedelta(hours=hour)).strftime('%Y/%m/%d %H:%M:%S'),
                'basal_hourly': round(basal_hourly, 3),
                'meal_bolus': meal_bolus
            })
    
    return pd.DataFrame(expanded_data)

def process_merged_data(merged_data):
    """处理合并后的数据"""
    if merged_data.empty:
        print("错误：未加载任何数据！")
        return None
        
    merged_data['rounded_time'] = merged_data['timestamp_original'].dt.round('H')
    
    merged_sorted = merged_data.sort_values(
        by=['item_id', 'rounded_time', 'timestamp_original'],
        ascending=[True, True, True]
    )
    
    merged_deduplicated = merged_sorted.drop_duplicates(
        subset=['item_id', 'rounded_time'], 
        keep='first'
    )
    
    final_df = merged_deduplicated.rename(columns={'rounded_time': 'timestamp'})
    final_df = final_df.drop(columns=['timestamp_original'], errors='ignore')
    
    keep_columns = [
        'item_id', 'timestamp', 'target', 
        'age', 'gender', 'weight', 
        'insulin', 'calories', 'drug'
    ]
    final_df = final_df[keep_columns]
    
    final_df['age'] = final_df['age'].astype('Int64')
    final_df['weight'] = final_df['weight'].astype(float)
    final_df['calories'] = final_df['calories'].astype(float)
    
    return final_df

def merge_insulin_data(merged_df, insulin_df):
    """合并胰岛素数据"""
    # 确保时间戳格式一致
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
    insulin_df['timestamp'] = pd.to_datetime(insulin_df['timestamp'])
    
    # 将时间戳规整到秒级
    merged_df['timestamp'] = merged_df['timestamp'].dt.floor('s')
    insulin_df['timestamp'] = insulin_df['timestamp'].dt.floor('s')
    
    # 合并数据
    merged = pd.merge(
        merged_df,
        insulin_df[['item_id', 'timestamp', 'basal_hourly', 'meal_bolus']],
        on=['item_id', 'timestamp'],
        how='left',
        validate='one_to_one'
    )
    
    # 使用basal_hourly更新insulin列，确保数值类型正确
    merged['basal_hourly'] = pd.to_numeric(merged['basal_hourly'], errors='coerce')
    merged['insulin'] = merged['basal_hourly']  # 直接赋值，不使用combine_first
    
    # 处理meal_bolus，确保数值类型正确
    merged['meal_bolus'] = (
        pd.to_numeric(merged['meal_bolus'], errors='coerce')
        .fillna(0.0)
        .astype(float)
        .round(1)
    )
    
    # 删除多余的列并确保数值列的类型
    result = merged.drop(columns=['basal_hourly'])
    result['insulin'] = result['insulin'].fillna(0.0).astype(float).round(3)
    
    # 打印一些调试信息
    print("\n胰岛素数据合并信息：")
    print(f"合并前insulin_df记录数：{len(insulin_df)}")
    print(f"合并后非空insulin值数量：{result['insulin'].notna().sum()}")
    print(f"insulin值范围：{result['insulin'].min()} - {result['insulin'].max()}")
    print(f"meal_bolus非零值数量：{(result['meal_bolus'] > 0).sum()}")
    
    return result

def main():
    """主处理流程"""
    # 1. 处理原始数据
    print("开始处理原始数据...")
    raw_df = process_original_data(DATA_PATH['raw'])
    if raw_df is None:
        print("原始数据处理失败，程序终止")
        return
        
    # 2. 添加患者元数据
    print("\n开始添加患者元数据...")
    raw_merged_file = os.path.join(DATA_PATH['processed'], 'merged_raw.csv')
    enriched_df = process_enriched_data(raw_merged_file)
    if enriched_df is None:
        print("元数据处理失败，程序终止")
        return
        
    # 3. 处理胰岛素数据
    print("\n处理胰岛素数据...")
    insulin_input = os.path.join(DATA_PATH['processed'], "insulin.csv")
    insulin_output = os.path.join(DATA_PATH['processed'], "processed_insulin_data.csv")
    insulin_df = process_insulin_data(insulin_input)
    insulin_df.to_csv(insulin_output, index=False)
    
    # 4. 合并所有数据
    print("\n开始合并所有数据...")
    final_output = os.path.join(DATA_PATH['processed'], "final_merged.csv")
    final_df = merge_insulin_data(enriched_df, insulin_df)
    final_df.to_csv(final_output, index=False)
    
    print("\n数据处理完成！")
    print(f"最终数据集信息：")
    print(f"总记录数：{len(final_df)}")
    print(f"时间范围：{final_df['timestamp'].min()} 至 {final_df['timestamp'].max()}")
    print("\n数据样例：")
    print(final_df.head(3))
    print("最终数据已保存到：", final_output)

if __name__ == "__main__":
    main() 