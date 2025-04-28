import pandas as pd
import json
import random
import os
from datetime import datetime, timedelta

# ========== 配置 ==========
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(WORKING_DIR, "../data/processed")
INPUT_CSV = os.path.join(DATA_PATH, "records.csv")
PATIENT_INFO_JSON = os.path.join(DATA_PATH, "patient_info.json")
OUTPUT_JSON = os.path.join(DATA_PATH, "training_pairs.json")

NUM_RECORDS = 16         # 每段输入历史长度
PREDICT_NEXT = 1         # 要预测的未来注射点数
WINDOW_STEP = 2          # 滑动窗口间隔（条数）
INJECTION_TIMES = ["00:00", "04:00", "09:00", "12:00", "17:00", "21:00"]

# ========== 数据加载 ==========
df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
df = df.sort_values(["patient_id", "timestamp"])

with open(PATIENT_INFO_JSON, "r", encoding="utf-8") as f:
    patient_info = json.load(f)

# ========== 时间映射函数 ==========
def round_to_injection_time(ts):
    current_time = ts.time()
    injection_candidates = [datetime.strptime(t, "%H:%M").time() for t in INJECTION_TIMES]
    nearest = min(injection_candidates, key=lambda t: abs(datetime.combine(datetime.today(), t) - datetime.combine(datetime.today(), current_time)))
    return nearest.strftime("%H:%M")

# ========== 样本生成 ==========
samples = []

for patient_id, group in df.groupby("patient_id"):
    group = group.reset_index(drop=True)
    patient_meta = patient_info.get(patient_id, {})

    total_possible = len(group) - NUM_RECORDS - PREDICT_NEXT
    if total_possible <= 0:
        continue

    for start_idx in range(0, total_possible, WINDOW_STEP):
        seq = group.iloc[start_idx:start_idx + NUM_RECORDS]
        future_candidates = group.iloc[start_idx + NUM_RECORDS:]

        future_points = []
        future_candidates = group.iloc[start_idx + NUM_RECORDS:]

        for pt in future_candidates.itertuples():
            time_str = pt.timestamp.strftime("%H:%M")
            if time_str in INJECTION_TIMES:
                future_points.append(pt)
            if len(future_points) == PREDICT_NEXT:
                break

        # 如果不足三条，跳过该样本
        if len(future_points) < PREDICT_NEXT:
            continue

        # 构造输入历史数据
        records = [
            {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "glucose": row["target"],
                "insulin": row["insulin"],
                "meal_bolus": row["meal_bolus"]
            }
            for _, row in seq.iterrows()
        ]

        # 构造患者信息块
        info_block = f"""患者信息:
                    - 年龄: {patient_meta.get("age", "未知")}岁, 体重: {patient_meta.get("weight", "未知")}kg, 性别: {patient_meta.get("gender", "未知")}
                    - 每日摄入热量: {patient_meta.get("calories", "未知")} kcal
                    - 病人的口服药物清单: {patient_meta.get("note", "无")}
                    """
        meds = patient_meta.get("medications", [])
        for i, m in enumerate(meds, 1):
            info_block += f"  {i}. {m}\n"

        # 构造 prompt
        prompt = (
            f"{info_block}\n"
            "字段说明：\n"
            "- timestamp: 时间（格式为 YYYY-MM-DD HH:MM）\n"
            "- glucose: 血糖值（单位：mmol/L）\n"
            "- insulin: 胰岛素注射速率（单位：Unit/h）\n"
            "- meal_bolus: 餐前摄入卡路里（单位：kcal）\n\n"
            "病人的临床数据如下：\n\n"
            + json.dumps(records, ensure_ascii=False, indent=2)
        )

        # 构造输出
        output_lines = []
        for pt in future_points:
            t = round_to_injection_time(pt.timestamp)
            dose = f"{pt.insulin:.2f}"
            output_lines.append(
                f"<next_injection_time>{t}</next_injection_time>\n"
                f"<recommended_dose>{dose} Unit/h</recommended_dose>"
            )

        output = "\n".join(output_lines)

        samples.append({
            "input": prompt,
            "output": output,
            "patient_id": patient_id
        })

# ========== 保存 ==========
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"✅ 成功生成训练对数：{len(samples)}，保存至 {OUTPUT_JSON}")
