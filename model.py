import torch
import torch.nn as nn

relu = torch.nn.ReLU()
# 创建 Tanh 层
tanh_layer = nn.Tanh()
class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 11  # changeable

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)  #给每一个学生进行嵌入
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)   #给每个题目的每个知识点都进行难度嵌入
        self.e_discrimination = nn.Embedding(self.exer_n, 1) #给每个题目一个区分度嵌入

        # 这里的三个全连接层的预测 应该是对应论文里面的三个交互函数
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization  初始化
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors  就是每个题目具体涉及了所有知识点中的哪些知识点 设计位置用1表示，不涉及的用0.0表示 注意 索引是从0开始
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))  #每个学生唯一的编号，映射成了学生对知识点的熟练度（掌握程度）
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10  #这里为什么要乘以10，我觉得只是为了扩大区分度的影响力 如果不乘以10  sigmod都是小于1的数 其实是缩小了区分度的影响力 和e_discrimination * (stu_emb - k_difficulty) 一起看？
        # prednet  认知诊断核心+三个交互函数
        input_x1 = e_discrimination * (stu_emb - k_difficulty) * kn_emb   #input_x相当于对当前题目的涉及知识点的把握度   训练完后的stu_emb相当于XX学生的能力值（在每个知识点的能力值，或熟练程度）
        input_x2 = self.drop_1(tanh_layer(self.prednet_full1(input_x1)))
        input_x3 = self.drop_2(tanh_layer(self.prednet_full2(input_x2)))
        # input_x3 = input_x1 + input_x3  #我自己的加的 飞哥的没有
        output = torch.sigmoid(self.prednet_full3(input_x3))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data



class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
