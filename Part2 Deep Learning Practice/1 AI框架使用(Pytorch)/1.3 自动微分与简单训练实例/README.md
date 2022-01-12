# 1.3 自动微分与简单训练实例

## 1.3.1 自动微分
在1.1当中已经引入过自动求导的相关代码实现。在深度学习框架当中，会根据我们设计的模型，系统会构建一个计算图（computational graph），
来跟踪计算是哪些数据通过哪些操作组合起来产生输出。 自动微分使系统能够随后反向传播梯度。意味着跟踪整个计算图，填充关于每个参数的偏导数。
下面我们通过pytorch来实现一个简单的实例：

```python
import torch

x = torch.arange(4.0)
x
```

```python
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值为None
```

计算$y = 2 X^TX$

```python
y = 2 * torch.dot(x, x)
y
```

下面通过调用反向传播函数来自动计算$y$关于$X$每个分量的梯度

```python
y.backward()
x.grad
```

显然结果和我们的数学推导$\frac{\partial y}{\partial X} = \frac{\partial (2X^TX)}{\partial X}
= 4X$是一致的

```python
x.grad == 4 * x  # True
```

当计算关于$X$的另一个函数的梯度时候，在默认情况下，PyTorch会累积梯度，我们需要使用```x.grad_zero_()```清除之前的值。

```python
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

当然，对**非标量的变量**也可以进行反向传播

```python
x.grad.zero_()
y = x * x
y
```

* 这里的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和
*
对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
在下面例子中，只想求偏导数的和，所以传递一个1的梯度是合适的:

```python
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

有时候，我们希望将某些计算移动到记录的计算图之外。
例如，假设$y$是作为$x$的函数计算的，而$z$则是作为$y$和$x$的函数计算的。我们想计算$z$关于$x$的梯度，但由于某种原因，我们希望将$y$视为一个常数，
并且只考虑到$x$在$y$被计算后发挥的作用。

下面例子在反向传播过程中将$u$当做一个常数进行处理：

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

简单验证一下：

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

即使在**控制流语句下**，梯度计算仍然可以正常工作：

$d = f(a) = k * a$

```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a  # 验证一下
```

## 1.3.2 简单训练实例

下面以简单线性回归为例子：

0. 生成数据集（一般情况下无需自己手动生成）

```python
import torch
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成数据
def synthetic_data(w, b, num_examples):
    """生成带噪音的数据集 y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

features, labels = synthetic_data(true_w, true_b, 1000)
```

1. 加载数据

```python
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 转成python的iter
next(iter(data_iter))
```

2. 定义模型

```python
# 模型定义
from torch import nn

# 单层神经网络
net = nn.Sequential(nn.Linear(2, 1))
```

3. 初始化参数

```python
# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

4. 定义损失函数和优化器

```python
# 损失函数
loss = nn.MSELoss()

# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

5. 开始训练

```python
# 开始训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 自带模型参数，不需要w和b放进去了
        trainer.zero_grad()  # 优化器梯度清零
        l.backward()  # 自动帮你求sum了
        trainer.step()  # 模型更新
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```
