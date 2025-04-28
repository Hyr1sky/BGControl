import os
import json

# ==== 配置路径 ====
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKING_DIR, "../data/processed")
INPUT_FILE = os.path.join(DATASET_DIR, "training_pairs.json")
DATA_JSON = os.path.join(DATASET_DIR, "data.json")
DATASET_INFO_JSON = os.path.join(DATASET_DIR, "dataset_info.json")

# ==== 构建默认指令和 system 提示 ====
DEFAULT_INSTRUCTION = "请仔细阅读病人的最新临床记录，仅根据所提供数据，预测下一次胰岛素注射的时间和推荐剂量。"
DEFAULT_SYSTEM = """你是一位医学AI助手，任务是根据糖尿病患者的最新血糖变化趋势，制定**下一次**胰岛素注射的时机和剂量。

你将接收到一段结构化的时间序列数据，字段说明如下：
- `timestamp`: 记录时间 (格式为 YYYY-MM-DD HH:MM)
- `glucose`: 血糖值 (单位：mg/dL)
- `insulin`: 胰岛素注射速率 (单位：Unit/h)
- `meal_bolus`: 进餐卡路里（单位：kcal）

**注射时间仅限以下六个固定时点**
[00:00, 04:00, 09:00, 12:00, 17:00, 21:00]

**预测任务说明**
- 请严格参考数据中**最后一个时间点之后**的下一个可注射时间点；
- 每次预测均应**仅参考当前输入内容**，不使用上下文中记忆的数据；
- 即便是同一位患者，每次输入不同，输出也应据此变化。

输出格式：

<next_injection_time>HH:MM</next_injection_time>  
<recommended_dose>X.XX Unit/h</recommended_dose>
"""

# ==== 读取原始数据 ====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ==== 转换为带 instruction 的格式 ====
converted = []
for item in raw_data:
    converted.append({
        "instruction": DEFAULT_INSTRUCTION,
        "input": item["input"],
        "output": item["output"],
        "system": DEFAULT_SYSTEM
    })

# ==== 保存转换后的 data.json ====
os.makedirs(DATASET_DIR, exist_ok=True)
with open(DATA_JSON, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成标准格式 data.json，共计样本数：{len(converted)}")

# ==== 构建 dataset_info.json ====
dataset_info = {
    "insulin_dosing": {
        "file_name": "data.json",
        "format": "alpaca",
        "columns": {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
            "system": "system"
        }
    }
}

# ==== 写入 dataset_info.json ====
with open(DATASET_INFO_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成 dataset_info.json 到: {DATASET_INFO_JSON}")
