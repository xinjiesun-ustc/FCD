####本文件的作用是：读取原始学生答题记录的csv文件，并转换成对应的数据，如：答错是0 部分对是1 全对是2。
####并保存对应的转换后的文件到磁盘


# 读取数据
import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = "./data/pisa-science-2015.csv"
df = pd.read_csv(file_path)

# 排除指定的字段
excluded_columns = ['STU_ID']
question_ids = [col for col in df.columns if col not in excluded_columns]

# 将答案转换为数值
def transform_to_number(ans):
    if isinstance(ans, str):  # 确保 ans 是字符串类型
        if 'No credit' in ans:
            return 0
        if 'Partial credit' in ans:
            return 0
        if 'Full credit' in ans:
            return 1
    return np.NaN  # 对于其他情况返回 NaN

# 应用转换
for q in question_ids:
    df[q] = df[q].map(transform_to_number)  # 映射到数字
    #     count_of_ones = (df[q] == 1).sum()  # 统计值为 1 的数量
    #     counts[q] = count_of_ones  # 将结果存储在字典中
    #
    # # 输出结果
    # # 计算所有问题中 1 的数量的总和
    # total_count_of_ones = sum(counts.values())
    #
    # # 输出总和
    # print(f"所有问题中 1 的数量的总和: {total_count_of_ones}")

# 保存映射后的 DataFrame 到 CSV 文件
output_file_path = "./data/pisa-science-2015-transform_to_number.csv"
df.to_csv(output_file_path, index=False)
print("transform_to_number转换保存成功！")




