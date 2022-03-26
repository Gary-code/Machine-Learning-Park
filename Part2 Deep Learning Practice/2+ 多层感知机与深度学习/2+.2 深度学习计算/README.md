## 2+.2 深度学习计算
> 此小节重点来讲解`Pytorch`中如何使用自定义的层与块。以及训练过程当中可做的优化。

```python
%matplotlib inline
import torch
from d2l import torch as d2l
from torch import nn
import torch.nn.functional as F
```

### 层和快

首先我们先来回顾一下多层感知机(MLP)

```python
net = nn.Sequential(nn.Linear(20, 256),
                    nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
net(X)
```

下面我们来自定义块:
* 重新定义上面的`MLP

```python
class MLP(nn.Module):
    """自定义块"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

net = MLP()
net(X)
```

* 自定义

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 20))

net(X)
```

* 自定义`forward`函数

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    # 做自己想做的事情
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)
```

* 混合其他组合块来使用

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

net = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
net(X)
```

自定义层:
* 无参数

```python
# 无参数自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X -X.sum()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

合并到更加复杂网络当中：

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

Y = net(torch.rand(4, 8))
Y.mean()
```

* 带参数

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, out_units))
        self.bias = nn.Parameter(torch.randn(out_units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

dense = MyLinear(5, 3)
dense.weight
```

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

### 参数管理

```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

X = torch.rand(size=(2,4))
net(X)
```

* 参数访问（拿出权重）

```python
print(net[2].state_dict())  # 最后输出层
```

* 目标参数

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

net[2].weight.grad == None
```

* 一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

* 通过名字获取参数

```python
net.state_dict()['2.bias'].data
```

* 嵌套块

```python
# 嵌套块
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```python
print(rgnet)
```

* 内置初始化
> 实际操作上不可以weight全部初始化为同一个数的

```python
# 内置初始化
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_weight)
net[0].weight.data[0], net[0].bias.data
```

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]  # 实际操作上不可以weight全部初始化为同一个数的
```

* 对某些块应用不同初始化方法, 使用$Xavier$初始化：

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data)
print(net[2].weight.data)
```

* 共享权重

```python
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
print(net[2].weight.data[0] == net[4].weight.data[0])
```

### 读写文件
> 本部分没有什么过度的技巧，具体通过理解代码来掌握`Pytorch`如何读写文件。

```python
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
x2
```

* 存放其他数据

```python
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

* 字典映射张量

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

* 加载和保存模型参数

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
x = torch.randn((2, 20))
Y = net(x)
# 保存
torch.save(net.state_dict(), 'mlp.params')

# 取参数（字典到tensor的映射）
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# 验证
Y_clone = clone(x)
Y_clone == Y
```

### 使用GPU
默认情况下都是在CPU上进行运算的。

我们先看看如何使用单个NVIDIA GPU进行计算。 首先，确保你至少安装了一个NVIDIA GPU。 然后，下载[NVIDIA驱动和CUDA](https://developer.nvidia.com/cuda-downloads) 并按照提示设置适当的路径。 当这些准备工作完成，就可以使用`nvidia-smi`命令来查看显卡信息。

**注意**:由于我的机子上是没有NVIDIA的GPU的，因此，运行下面的代码会报错。

```python
!nvidia-smi
```

```python
torch.device('cpu')
torch.cuda.device('cuda')  # 第0个GPU
torch.cuda.device('cuda:1')  # 第1个GPU

# 查询可用gpu的数量
torch.cuda.device_count()

# 根据索引号查看GPU名字:
torch.cuda.get_device_name(0)


# 允许我们在请求GPU不存在的时候用CPU来跑代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i), 否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpu():
    """ 返回所有可用GPU，没有可用就返回CPU"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpu()
```

* 在GPU上闯将Tensor

```python
x = torch.tensor([1, 2, 3])
x.device  # CPU上

# 存储在GPU上
X = torch.ones(2, 3, device=try_gpu())
X

# 第二个GPU上创建一个随机张量
Y = torch.rand(2, 3, device=try_gpu(1))
Y

# 运算, 要在同一个GPU上, GPU向CPU传数据非常慢的
Z = X.cuda(1)
print(X)
print(Z)
Y + Z
Z.cuda(1) is Z  # True
```

* 将神经网络模型放入GPU

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

net(X)

# 确认模型参数存储在同一个GPU上
net[0].weight.data.device
```
