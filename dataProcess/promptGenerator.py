import pandas as pd
import json
import os
from tqdm import tqdm

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_DIR, "../data/processed")

INPUT_FILE = os.path.join(DATA_PATH, "final_merged.csv")
OUTPUT_FILE = os.path.join(DATA_PATH, "prompt.json")

INJECTION_TIMES = ["00:00", "04:00", "09:00", "12:00", "17:00", "21:00"]

df = pd.read_csv(INPUT_FILE)

# 检查所需列是否存在
required_cols = {"item_id", "timestamp", "target", "insulin", "meal_bolus"}
if not required_cols.issubset(set(df.columns)):
    raise ValueError(f"Missing required columns, expected to contain: {required_cols}")

# insulin 为 0, 记录不全
df = df[df["insulin"]!= 0.00]

# 将时间戳字符串转为时间格式（可选排序）
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["item_id", "timestamp"])

# 按病人分组生成 prompt
prompts = []
records_csv = []

for patient_id, group in tqdm(df.groupby("item_id")):
    records = []
    for _, row in group.iterrows():
        time_str = row["timestamp"].strftime("%H:%M")
        insulin_value = row["insulin"] if time_str in INJECTION_TIMES else 0.0

        record = {
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "glucose": row["target"],
            "insulin": insulin_value,
            "meal_bolus": row["meal_bolus"]
        }
        records.append(record)
        records_csv.append(f"{row['item_id']},{row['timestamp']},{row['target']},{insulin_value},{row['meal_bolus']}")

    prompt = {
        "patient_id": patient_id,
        "prompt": f"以下是患者 {patient_id} 的血糖记录，请分析是否需要调整胰岛素剂量：\n\n{json.dumps(records, indent=2, ensure_ascii=False)}\n\n请给出建议: 下一个时刻胰岛素剂量应为多少？"
    }
    prompts.append(prompt)

# 保存为 JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(prompts)} prompts and saved to {OUTPUT_FILE}")

# 保存为 CSV
with open(os.path.join(DATA_PATH, "records.csv"), "w", encoding="utf-8") as f:
    f.write("patient_id,timestamp,target,insulin,meal_bolus\n")
    f.write("\n".join(records_csv))

print(f"✅ Generated {len(records_csv)} records and saved to {os.path.join(DATA_PATH, 'records.csv')}")