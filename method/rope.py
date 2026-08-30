import torch

x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([10, 20, 30, 40, 50])

condition = x > 3

result = torch.where(condition, x, y)
print(result)


a = torch.arange(0, 10, 2)
b = torch.arange(-5, 0, 1)
print(a)
print(b)

v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
v = torch.outer(v1, v2)
print(v)

t1 = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]])
t2 = torch.tensor([[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]])
result1 = torch.cat((t1, t2), dim=0)
result2 = torch.cat((t1, t2), dim=1)
result3 = torch.cat((t1, t2), dim=2)
print(result1)
print(result2)
print(result3)
