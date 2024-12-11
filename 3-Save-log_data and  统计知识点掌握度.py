import numpy as np
import pandas as pd
import json
from tqdm import tqdm  # 导入 tqdm 库
import random

# 读取所有题目的数据
file_path_questions = "./data/pisa-science-2015-transform_to_number.csv"
df = pd.read_csv(file_path_questions, low_memory=False)

# 读取需要的题目 ID
file_path_ids = "./data/pisa2015-science-kcs.csv"
df_ids = pd.read_csv(file_path_ids, low_memory=False)

# 提取需要的题目 ID 列
required_ids = df_ids['ID'].unique()

# 排除指定的字段
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# 筛选出所需题目
filtered_question_ids = [qid for qid in question_ids if qid in required_ids]

# 确保包括 STU_ID 列和筛选后的题目列
df_new = df.loc[:, ['STU_ID'] + filtered_question_ids]

# 清理数据，删除没有有效回答的列
df_question = df_new.loc[:, filtered_question_ids]
column_counts = df_question.count()
columns_with_zero_count = column_counts[column_counts == 0].index
df_question_cleaned = df_question.drop(columns=columns_with_zero_count)

# 筛选答题数量大于或等于 20 条有效记录的学生
df_new['Question Count'] = df_question_cleaned.notnull().sum(axis=1)
df_student_filtered = df_new[df_new['Question Count'] >= 30].copy()  #science read  30 math 20

# 加载 exer_id_mapping.json
with open('./data/exer_id_mapping.json', 'r') as file:
    exer_id_mapping = json.load(file)

# 加载 Q_matrix
Q_matrix = np.load('./data/Q_matrix.npy')

# 重新映射 STU_ID
stu_id_mapping = {stu_id: i + 1 for i, stu_id in enumerate(df_student_filtered['STU_ID'].unique())}
df_student_filtered['STU_ID'] = df_student_filtered['STU_ID'].map(stu_id_mapping)

num_knowledge_points = Q_matrix.shape[1]
# 初始化学生和知识点掌握情况的字典
student_knowledge = {int(stu_id): np.zeros(Q_matrix.shape[1]) for stu_id in df_student_filtered['STU_ID']}
student_knowledge_counts = {int(stu_id): np.zeros(Q_matrix.shape[1]) for stu_id in   df_student_filtered['STU_ID']}  # 每个学生对每个知识点的题目计数

knowledge_appearances = np.zeros(Q_matrix.shape[1])  # 每个知识点在题目中的累计出现次数
knowledge_totals = np.zeros(num_knowledge_points)  # 每个知识点的总掌握度

# 计算每个学生的每个知识点掌握情况
for index, row in tqdm(df_student_filtered.iterrows(), total=df_student_filtered.shape[0], desc="Processing Students"):
    user_id = int(row['STU_ID'])

    for column in filtered_question_ids:
        exer_id = exer_id_mapping.get(column)  # 获取题目编号
        score = row[column]  # 得分为 0 或 1 或 2

        if exer_id is not None and not np.isnan(score):
            # 获取题目所包含的知识点
            knowledge_points = np.where(Q_matrix[exer_id - 1] > 0)[0]
            # 更新学生知识点掌握情况
            student_knowledge[user_id][knowledge_points] += score
            # 更新全体学生的知识点总得分
            knowledge_totals[knowledge_points] += score
            # 记录该学生在当前题目中知识点的出现次数
            student_knowledge_counts[user_id][knowledge_points] += 1
            # 更新知识点在题目中的累计出现次数
            knowledge_appearances[knowledge_points] += 1

# 计算每个学生的知识掌握度
for user_id in tqdm(student_knowledge.keys(), desc="Calculating Student Knowledge Mastery"):
    user_scores = student_knowledge[user_id]
    counts = student_knowledge_counts[user_id]  # 当前学生对每个知识点的题目计数
    # 计算掌握度
    student_knowledge[user_id] = np.divide(user_scores, np.maximum(counts, 1), out=np.zeros_like(user_scores))

# 计算整体掌握情况，按知识点得分除以知识点累计出现次数
overall_knowledge = (knowledge_totals / np.maximum(knowledge_appearances, 1)).tolist()

#
# # 保存到 JSON 文件
# output_data = {
#     "student_knowledge": {user_id: user_scores.tolist() for user_id, user_scores in student_knowledge.items()},
#     "overall_knowledge": overall_knowledge  # 这已经是列表，不需要转换
# }

# 计算每个学生相对于整体的知识掌握差异
student_mastery_difference = {
    user_id: (user_scores - overall_knowledge).tolist()
    for user_id, user_scores in student_knowledge.items()
}

# 将差异结果添加到 output_data 中
output_data = {
    "student_knowledge": {user_id: user_scores.tolist() for user_id, user_scores in student_knowledge.items()},
    "overall_knowledge": overall_knowledge,  # 已经是列表格式
    "student_knowledge_difference": student_mastery_difference  #学生个人的知识点数量度-整体的该知识点熟练度
}

# 输出到 JSON 文件
import json
with open("output_data.json", "w") as f:
    json.dump(output_data, f)

with open('./data/knowledge_mastery.json', 'w') as file:
    json.dump(output_data, file, indent=4)

print("统计在知识点的个人和整体掌握度处理完成，数据已保存为 JSON 格式。")

#保存log_data
# 处理每一行数据
logs = []
for index, row in df_student_filtered.iterrows():
    user_id = row['STU_ID']
    user_logs = []
    valid_answer_count = 0  # 计数有效答案

    for column in filtered_question_ids:  # 只处理题目相关列
        exer_id = exer_id_mapping.get(column)  # 获取题目编号
        score = row[column]  # 已经是数字

        if exer_id is not None and not np.isnan(score):
            # 获取知识点
            knowledge_points = np.where(Q_matrix[exer_id - 1] > 0)[0] + 1  # 获取知识点编号
            user_logs.append({
                "exer_id": exer_id,
                "score": score,
                "knowledge_code": knowledge_points.tolist()  # 转换为列表
            })
            # if score in [0, 1, 2]:  # 计算有效的答题个数
            valid_answer_count += 1

    logs.append({
        "user_id": int(user_id),
        "log_num": valid_answer_count,  # 有效答题个数
        "logs": user_logs
    })

# 保存到 JSON 文件
with open('./data/log_data.json', 'w') as outfile:
    json.dump(logs, outfile, indent=4)

print("log_data处理完成，数据已保存为 JSON 格式。")