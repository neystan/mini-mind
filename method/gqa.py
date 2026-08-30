import torch
import torch.nn as nn

# Dropout是指随机丢弃某些值，并保持期望不变，其它部分扩大两倍
# dropout_layer = nn.Dropout(p=0.5)

# t1 = torch.Tensor([1, 2, 3])
# t2 = dropout_layer(t1)

# print(t2)


# 进行线性变化， 就是对应用的张量 乘以 一个 w 矩阵 + b（这里的w、b都是随机的）
# layer = nn.Linear(in_features = 3, out_features = 5, bias = True)

# t1 = torch.Tensor([1, 2, 3])
# t2 = torch.Tensor([[1, 2, 3]])

# output1 = layer(t1)
# output2 = layer(t2)
# print(output1)
# print(output2)


# # 改变张量的形状
# t = torch.Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
# t1_view = t.view(3, 4)
# t2_view = t.view(4, 3)
# print(t1_view)
# print(t2_view)


# # 交换维度
# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# t1_transpose = t1.transpose(0, 1)
# print(t1_transpose)


# # 对角线下数字置0
# t1 = torch.tensor([[1, 2, 3], [4, 5, 6], [13, 14, 15]])

# print(torch.triu(t1))
# print(torch.triu(t1, diagonal=1))


# # 改变张量形状
# t1 = torch.arange(1, 7, 1)
# t1_reshape = t1.reshape(2, 3)
# print(t1_reshape)

