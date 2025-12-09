import torch
from main import test
hidden_layer = int(input())

# 任务：使用 torch 模块构建一个简单神经网络模型
# ********** Begin *********#

batch_n = 100     # 在一个批次中输入数据的数量
hidden_layer = 100     # 定义经过隐藏层后保留的数据特征的个数
input_data = 1000     # 数据包含的数据特征个数
output_data = 10     # 输出的数据，可将输出的数据看作一个分类结果值的数量


# ********** End **********#
 
x = torch.randn(batch_n, input_data)
y = torch.randn(batch_n, output_data)
 
w1 = torch.randn(input_data, hidden_layer)
w2 = torch.randn(hidden_layer, output_data)
 
epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    h1 = x.mm(w1)  
    h1 = h1.clamp(min=0)
    y_pred = h1.mm(w2)  
    loss = (y_pred - y).pow(2).sum()
    gray_y_pred = 2 * (y_pred - y)
    gray_w2 = h1.t().mm(gray_y_pred)
 
    grad_h = gray_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)
 
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * gray_w2

test()