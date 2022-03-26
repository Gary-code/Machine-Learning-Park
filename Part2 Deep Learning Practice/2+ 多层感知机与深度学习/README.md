# 2+ 多层感知机与深度学习

> 2+章节主要作为2章节的补充，介绍在Pytorch中如何使用深度学习的相关技术。

## 2+.1 多层感知机(MLP)与训练

### 模型启发

感知机模型:

* 感知机是一个二分类模型，是最早的Al模型之一
* 它的求解算法等价于使用批量大小为1的梯度下降
* 它不能拟合XOR函数，导致的第一次AI寒冬。（只能产生线性分割面）

为此，人们后来引入多层感知机来解决**XOR**的问题。
![image-20211015202057316](https://s2.loli.net/2022/03/25/zaGI8cY9Jt6BAwd.png)

下面是**单隐藏层**的MLP示意图:
![image-20211015202124078](https://s2.loli.net/2022/03/25/Zf7JbiW92t6C3Bn.png)


### 激活函数

引入非线性的激活函数，其公式如下（$\sigma$为按元素的激活函数）：

$$
输入 \mathbf{x} \in \mathbb{R}^{n} \\
隐藏层 \mathbf{W}_{1} \in \mathbb{R}^{m \times n}, \mathbf{b}_{1} \in \mathbb{R}^{m}  \\
输出层 \mathbf{w}_{2} \in \mathbb{R}^{m}, b_{2} \in \mathbb{R}
$$

$$
\begin{aligned}
&\mathbf{h}=\sigma\left(\mathbf{W}_{1} \mathbf{x}+\mathbf{b}_{1}\right) \\
&o=\mathbf{w}_{2}^{T} \mathbf{h}+b_{2}
\end{aligned}
$$
下面我们通过`python`代码将几种常见的激活函数画出来:

```python
%matplotlib inline
import torch
from d2l import torch as d2l
```

* ReLU

简单好用:
$$\textbf{ReLU}(x) = max(x, 0)$$

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))  # detach() 去掉梯度
```

下面绘制ReLU的导数:

```python
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

可以看到，使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题

* sigmoid

(0, 1)之间:
$$\operatorname{sigmoid}(x)=\frac{1}{1+\exp (-x)}$$

```python
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

其导数为:

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

当输入为0时，sigmoid函数的导数达到最大值0.25； 而输入在任一方向上越远离0点时，导数越接近0。

* tanh

当输入在0附近时，tanh函数接近线性变换。 函数的形状类似于sigmoid函数， 不同的是tanh函数关于坐标系原点中心对称：
$$\tanh (x)=\frac{1-\exp (-2 x)}{1+\exp (-2 x)}$$

```python
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

其导数为：

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

 当输入接近0时，tanh函数的导数接近最大值1。 与我们在sigmoid函数图像中看到的类似， 输入在任一方向上越远离0点，导数越接近0。

### 多层感知机代码实现

通过上面的介绍，相信对多层感知机有了一定的了解。下面我们运用之前学到过的知识来调佣`Pytorch`的`API`实现多层感知机。

```python
import torch
from torch import nn
from d2l import torch as d2l
```

* 定义模型

```python
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

def init_weights(m):
    """
    初始化模型参数
    :param m:
    :return:
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

* 开始训练：
  由于计算资源优先，这里不做训练，感兴趣的读者可以将`d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)`取消注释查看训练效果。

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 训练
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

### $Xavier$初始

在上面的部分，我们使用正态分布来初始化权重值。如果我们不指定初始化方法， 框架将使用默认的随机初始化方法，对于中等难度的问题，这种方法通常很有效。

* 在合理值区间里随机初始参数
* 训练开始的时候更容易有数值不稳定
  * 远离最优解的地方损失函数表面可能很复杂
  * 最优解附近表面会比较平
* 使用$N$(0, 0.01)来初始可能对小网络没问题，但不能保证深度神经网络


* **期望**

  > **均值均为0**

![image-20220325114621260](https://s2.loli.net/2022/03/25/xvRjoNsUhm7QIyq.png)

* **方差**

> 输入和输出的方差一样
>
> 所以$n_{t-1} \gamma_{t}=1$
>
> * 因为 $h^t$ 是由 $t-1$ 层的 $n$ 个参数 $w$ 运算求得的，而 t-1 层的这些参数之前假设了他们都是服从方差为 $\gamma$ 的分布，所以他们相加就成了 $n_{t-1}\gamma$, 了。
>
> ![image-20211102191920125](C:\Users\Gary\AppData\Roaming\Typora\typora-user-images\image-20211102191920125.png)

$$
\begin{aligned}
\operatorname{Var}\left[h_{i}^{t}\right] &=\mathbb{E}\left[\left(h_{i}^{t}\right)^{2}\right]-\mathbb{E}\left[h_{i}^{t}\right]^{2}=\mathbb{E}\left[\left(\sum_{j} w_{i, j}^{t} h_{j}^{t-1}\right)^{2}\right] \\
&=\mathbb{E}\left[\sum_{j}\left(w_{i, j}^{l}\right)^{2}\left(h_{j}^{t-1}\right)^{2}+\sum_{j \neq k} w_{i, j}^{l} w_{i, k}^{t} h_{j}^{t-1} h_{k}^{t-1}\right] \\
&=\sum_{j} \mathbb{E}\left[\left(w_{i, j}^{l}\right)^{2}\right] \mathbb{E}\left[\left(h_{j}^{t-1}\right)^{2}\right] \\
&=\sum_{j} \operatorname{Var}\left[w_{i, j}^{t}\right] \operatorname{Var}\left[h_{j}^{t-1}\right]=n_{t-1} \gamma_{t} \operatorname{Var}\left[h_{j}^{t-1}\right]
\end{aligned}
$$

$n_{t-1} \gamma_{t}=1$



* **反向均值和方差**

> * 均值为0
> * 方差一样

$$
\begin{aligned}
&\frac{\partial \ell}{\partial \mathbf{h}^{t-1}}=\frac{\partial \ell}{\partial \mathbf{h}^{t}} \mathbf{W}^{t} =>\quad\left(\frac{\partial \ell}{\partial \mathbf{h}^{t-1}}\right)^{T}=\left(W^{t}\right)^{T}\left(\frac{\partial \ell}{\partial \mathbf{h}^{t}}\right)^{T} \\
&\mathbb{E}\left[\frac{\partial \ell}{\partial h_{i}^{t-1}}\right]=0 \\
&\operatorname{Var}\left[\frac{\partial \ell}{\partial h_{i}^{t-1}}\right]=n_{t} \gamma_{t} \operatorname{Var}\left[\frac{\partial \ell}{\partial h_{j}^{t}}\right] \quad=>n_{t} \gamma_{t}=1
\end{aligned}
$$



* $Xavier$初始化

> 输入输出很难控制的

![image-20220325114548076](https://s2.loli.net/2022/03/25/nvOiNbFwpjIMXl2.png)

$Xavier$初始化代码如下:

```python
import math

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=math.sqrt(2/(m.in_features+m.out_features)))
net.apply(init_weights)
```

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