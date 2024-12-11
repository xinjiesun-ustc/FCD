####本文件的作用是：生成Q矩阵，知识点映射、题目ID映射、题目和知识点的关系映射

import pandas as pd
import numpy as np
import json

# 读取 CSV 文件，尝试不同的编码
file_path = "./data/pisa2015-science-kcs.csv"
kc_list = []
exercises_data = {}

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # 或者尝试 'utf-16'

    # 创建 ID 映射字典
    exer_id_mapping = {row['ID']: i + 1 for i, row in df.iterrows()}  # ID 从1开始

    # 提取所有唯一的知识点
    for column in df.columns:
        if column != 'ID':
            unique_values = df[column].unique()  # 获取每一列的唯一值
            kc_list.extend(unique_values)  # 直接添加到 kc_list

    # 创建知识点的唯一集合并映射
    unique_kc = set(kc_list)
    knowledge_points_map = {kc: i for i, kc in enumerate(sorted(unique_kc))}

    # 存储每个题目的知识点
    for index, row in df.iterrows():
        exer_id = exer_id_mapping[row['ID']]
        knowledge_list = []
        for column in df.columns:
            if column != 'ID':
                if row[column] in knowledge_points_map:
                    knowledge_list.append(row[column])
        exercises_data[exer_id] = knowledge_list

    # 只保留 kc_list 中具体数据
    kc_list = list(unique_kc)
    print("知识点KC映射字典是:", knowledge_points_map)
    print("知识点KC是:", kc_list)
    print("知识点KC有:", len(kc_list))

except Exception as e:
    print(f"发生错误: {e}")

# 构建 Q 矩阵
num_exercises = len(exercises_data)
num_knowledge_points = len(knowledge_points_map)

# 初始化 Q 矩阵
Q_matrix = np.zeros((num_exercises, num_knowledge_points))

# 填充 Q 矩阵
for i, (exer_id, knowledge_list) in enumerate(exercises_data.items()):
    for kc in knowledge_list:
        if kc in knowledge_points_map:
            Q_matrix[i, knowledge_points_map[kc]] = 1

# 保存 Q 矩阵和 ID 映射到文件
np.save('./data/Q_matrix.npy', Q_matrix)
np.save('./data/exercises_data.npy', exercises_data)

# 保存 ID 映射为 JSON 文件
with open('./data/exer_id_mapping.json', 'w') as f:
    json.dump(exer_id_mapping, f)


# 保存KC 映射为 JSON 文件
with open('./data/knowledge_points_map.json', 'w') as f:
    json.dump(knowledge_points_map, f)
print("Q 矩阵和 ID 映射已保存。")
