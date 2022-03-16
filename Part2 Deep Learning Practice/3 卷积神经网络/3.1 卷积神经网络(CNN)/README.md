## 3.1 卷积神经网络(CNN)
之前讨论的多层感知机十分适合处理表格数据，其中行对应样本，列对应特征。
对于表格数据，寻找的模式可能涉及特征之间的交互，但是不能预先假设任何与特征交互相关的先验结构。
此时，多层感知机可能是最好的选择，然而对于高维感知数据，这种缺少结构的网络可能会变得不实用。

> example
在猫狗分类的例子中：假设我们有一个足够充分的照片数据集，数据集中是拥有标注的照片，每张照片具有百万级像素，这意味着网络的每次输入都有一百万个维度。
即使将隐藏层维度降低到1000，这个全连接层也将有$10^6×10^3=10^9$个参数。想要训练这个模型将不可实现，因为需要有大量的GPU、分布式优化训练的经验和极长时间的等待。
* RGB图片有36M元素(3个channel)
* 使用100大小的单隐藏层的MLP，模型有3.6B元素
    * **远远多于世界上所有猫和狗的总和！**
* **14GB的内存！！！**
    * 相当消耗GPU！


为此，我们引入卷积层来解决这些问题，其遵循两个**重要原则**:
* **平移不变形**
* 不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
* **局部性**
    *
神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系.

**关于卷积**：
在数学中，两个函数（比如$f, g:
\mathbb{R}^d \to \mathbb{R}$）之间的“卷积”被定义为

$$(f * g)(\mathbf{x}) = \int
f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$
也就是说，卷积是当把一个函数“翻转”并移位$\mathbf{x}$时，测量$f$和$g$之间的重叠。
当为离散对象时，积分就变成求和。例如：对于由索引为$\mathbb{Z}$的、平方可和的、无限维向量集合中抽取的向量，我们得到以下定义：

$$(f *
g)(i) = \sum_a f(a) g(i-a).$$

对于二维张量，则为$f$的索引$(a, b)$和$g$的索引$(i-a,
j-b)$上的对应**加和**：

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
需要**注意的是**:卷积神经网络是为了影像(*image*)而设计的，为此在CNN迁移到其他的领域应用时，应该考虑该领域与影像是否有相似的地方。

### 3.1.1 全连接层 => 卷积层

#### 平移不变性
现在我们输入输出看为一个矩阵的形式$(宽度, 高度)$,
将权重矩阵变形为*4-D*张量为$(h, w) => (h', w')$
$$h_{i, j}=\sum_{k, l} w_{i, j, k, l} x_{k,
l}=\sum_{a, b} v_{i, j, a, b} x_{i+a, j+b}$$
其中$w$为全连接层的权重， $x$是输入。$v_{i, j, a,
b}=w_{i, j, i+a, j+b}$，$V$是$W$的重新索引

$x$的平移导致$h$的平移: $$h_{i, j}=\sum_{a, b}
v_{i, j, a, b} x_{i+a, j+b}$$

而根据平移不变性原则，$v$不应该依赖于$(i,
j)$，为此我们使用二维交叉相关的方式(卷积),令$v_{i, j, a, b}=v_{a, b}$:
$$h_{i, j}=\sum_{a, b} v_{a,
b} x_{i+a, j+b}$$


#### 局部性
根据
$$h_{i, j}=\sum_{a, b} v_{a, b} x_{i+a, j+b}$$
评估$h_{i,j}$时，不应该用远离$x_{i,j}$的参数，因此，当$|a|,|b|>\Delta$时，使得$v_{a,b}=0$，即
$$h_{i,
j}=\sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} v_{a, b} x_{i+a, j+b}$$
简单的来说，运用两个原则就可以将全连接层转化为卷积层(CNN):
$$\begin{aligned}
&h_{i, j}=\sum_{a, b} v_{i,
j, a, b} x_{i+a, j+b}
=> &h_{i, j}=\sum_{a=-\Delta}^{\Delta}
\sum_{b=-\Delta}^{\Delta} v_{a, b} x_{i+a, j+b}
\end{aligned}$$

### 3.1.2 二维交叉相关(卷积)
我们定义kernel(核)就是上面所说的$w$，计算过程如下图所示:
![](https://zh-v2.d2l.ai/_images/correlation.svg)

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

**需要注意的是**:
输出大小略小于输入大小。这是因为卷积核的宽度和高度大于1，而卷积核只与图像中每个大小完全适合的位置进行互相关运算。
所以，输出大小等于输入大小$n_h
\times n_w$减去卷积核大小$k_h \times k_w$，即：

$$(n_h-k_h+1) \times (n_w-k_w+1)$$
这是因为需要足够的空间在图像上“移动”卷积核。
稍后，将看到如何通过在图像边界周围填充零来保证有足够的空间移动卷积核，从而保持输出大小不变。接下来，在`corr2d`函数中实现如上过程，该函数接受输入张量`X`和卷积核张量`K`，并返回输出张量`Y`。
同样的，扩展到三维的计算也和上述过程类似，但由于卷积层是为图像问题设计的，所以一般情况下使用的都是二维卷积。

*
**卷积层和输入和核矩阵进行交叉相关，加上偏移后得到输出**
* 核矩阵和偏移是可学习的参数
* **核矩阵的大小就是超参数（控制了局部性）**
下面我们通过*从零开始实现*和*调用Pytorch API 简约实现的方式*来进行实践:

```python
import torch
from torch import nn
```

* 由零开始实现

```python
def corr2d(X, K):
    """计算二维卷积"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1), (X.shape[1] - w + 1))  # 输出结果
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
K = torch.tensor([[0, 1], [2, 3]])
corr2d(X, K)
```

定义卷积层

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
```

不同颜色边缘检测

```python
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
Y
```

* 调用Pytorch 接口简约实现

下面展示对超参数$K$的学习：

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)  # 第一个和第二个参数表示输入输出通道都为1（黑白为1，彩色为3）

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad  # 裸写梯度下降
    print(f'batch: {i}, loss: {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))
```

从上面例子可以看出来，目前所讲解的卷积层所需要学习的参数就是$K$

### 3.1.3 填充和步幅

填充(padding)和步幅(stride)是控制卷积层输出大小的两个超参数。

```python
import torch
from torch import nn

def comp_conv2d(conv2d, X):
    """考虑通道数的四维卷积"""
    X = X.reshape((1, 1) + X.shape)  # 通道数
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 四维东西，把前面两维拿掉
```

#### 填充(padding)
通过输出大小$(n_h - k_h + 1) \times (n_w - k_w +
1)$可知，更大的卷积核可以*更快的减小输出的大小*。因此我们需要引入填充技术减缓输出的减小, 如下图所示，在输入周围填充0：
![image-20211106172347979](https://s2.loli.net/2022/01/28/BAaUceZNVu5m2td.png)
填充$p_h$行和$p_w$列，输出的形状为:
$$\left(n_{h}-k_{h}+p_{h}+1\right)
\times\left(n_{w}-k_{w}+p_{w}+1\right)$$

可以令输入和输出的形状保持一致

通常$p_h = k_h
-1$,$p_w=k_w-1$
* 当$k_h$为奇数：上下填充$p_h/2$
* 当$k_h$为偶数：上侧填充$\left\lceil p_{h} /
2\right\rceil$,下侧填充$\left\lfloor p_{h} / 2\right\rfloor$

```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # padding 为填充,左右各一列
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape  # 8 * 8
```

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))  # 上下为2，左右为1
comp_conv2d(conv2d, X).shape  # 8 * 8
```

#### 步幅(stride)
有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。滑动窗口的个数就是步幅。
下面是垂直步幅为$3$，水平步幅为$2$的二维互相关运算。
着色部分是输出元素以及用于输出计算的输入和内核张量元素：$0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$。
![](https://zh-v2.d2l.ai/_images/conv-stride.svg)
可以看到，为了计算输出中第一列的第二个元素和第一行的第二个元素，卷积窗口分别向下滑动三行和向右滑动两列。但是，当卷积窗口继续向右滑动两列时，没有输出，因为输入元素无法填充窗口（除非添加另一列填充）。
通常，当垂直步幅为$s_h$、水平步幅为$s_w$时，输出形状为

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times
\lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$
如果设置了$p_h=k_h-1$和$p_w=k_w-1$，则输出形状将简化为$\lfloor(n_h+s_h-1)/s_h\rfloor \times
\lfloor(n_w+s_w-1)/s_w\rfloor$。
更进一步，如果输入的高度和宽度可以被垂直和水平步幅整除，则输出形状将为$(n_h/s_h)
\times (n_w/s_w)$。

下面我们展示Pytorch的相关实现:

```python
# 步幅设置为2(w和h)
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)

comp_conv2d(conv2d, X).shape
```

```python
# 稍微复杂一点例子
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

### 3.1.4 输入输出通道(channel)
> 卷积神经网络非常重要的超参数

对于*黑白图片*而言这有一个通道，而对于*彩色图片*来说，有(R, G,
B)三个通道，若将彩色图片转化成灰度图会丢失很多信息，因此我们很多时候都要基于三通道图片作为输入来进行处理。

采用的处理方法如下:
*
**每个通道都有对应的卷积核**
* **各自做完卷积核再相加**
![](https://s2.loli.net/2022/01/28/IA6BE875XusxKZc.png)
**关于Pytorch简单实现**，上面的例子都进行了讲解，即
```python
conv2d = nn.Conv2d(input_channels,
output_channels, kernel_size, padding, stride)
```
下面的代码实践都是从零开始的实现。

#### 多个输入通道
* 每一个通道都有一个卷积核，结果是所有通道的卷积结果再相加，如下图所示
![](https://s2.loli.net/2022/01/31/GPmbZc1YD35tJLI.png)

$$\begin{aligned}
&(1
\times 1+2 \times 2+4 \times 3+5 \times 4)
&+(0 \times 0+1 \times 1+3 \times 2+4
\times 3)=56
\end{aligned}$$

具体使用公式表达如下:
* 输入: $\mathbf{X}: c_{i} \times n_{h}
\times n_{w}$
* 核: $\mathbf{W}: c_o \times c_i \times k_h \times k_w$
* 偏差:
$\mathbf{B}: c_o \times c_i$
* 输出: $\mathbf{Y}: c_o \times m_h \times m_w$
*
计算复杂度(浮点计算数FLOP):$O\left(c_{i} c_{o} k_{h} k_{w} m_{h} m_{w}\right)$
$$\mathbf{Y}=\mathbf{X} \star \mathbf{W}+\mathbf{B}$$

下面我们通过代码进行实践:

```python
import torch

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

#### 多输出通道
我们可以有多个三维卷积核，每个核生成一个输出通道
$$\mathbf{Y}_{i,:,:}=\mathbf{X} \star
\mathbf{W}_{i,:,:,:} \text { for } i=1, \ldots, c_{o}$$

下面进行代码实践。

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0)
K.shape
```

#### 为什么要多个输入输出通道
* 每个*输出通道*可以识别**特定模式**
* *输入通道*核识别并**组合输入**中的模式


#### $ 1 \times 1 $ 卷积层
$1 \times 1$卷积，即$k_h = k_w = 1$，看起来似乎没有多大意义。实际上，其在很多工程中相当流行。
因为使用了最小窗口，$1\times
1$卷积失去了卷积层的特有能力——在高度和宽度维度上，识别**相邻元素间相互作用**的能力。其唯一的计算发生在**通道上**。
![](https://zh-v2.d2l.ai/_images/conv-1x1.svg)

图中，卷积核上方为输出0的通道，下方为输出为1的通道。
上图展示了使用$1 \times 1$卷积核与3个输入通道和2个输出通道的卷积计算。
这里输入和输出具有相同的高度和宽度，输出中的每个元素都是从输入图像中同一位置的元素的线性组合。可以将$1 \times
1$卷积层看作是在每个像素位置应用的全连接层（形状为$n_hn_w \times c_i$,权重为$c_o \times
c_i$的全连接层），以$c_i$个输入值转换为$c_o$个输出值。 因为这仍然是一个卷积层，所以跨像素的权重是一致的。 同时，$1 \times
1$卷积层需要的权重维度为$c_o \times c_i$，再额外加上一个偏置。简单来说:
* 它不识别空间模式
* 作用是融合通道
下面，使用全连接层实现$1 \times 1$卷积。 注意，需要对输入和输出的数据形状进行调整。

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

float(torch.abs(Y1 - Y2).sum()) < 1e-6  # True
```

### 3.1.5 池化层(Pooling Layers)

卷积层对位置信息相当敏感：
*
正如我们之间说到的边缘检测的例子，如果边缘线沿着直线有小小偏差（1像素位移）就会导致0输出，这显然不是我们想要看到的结果。
因此我们需要一定程度的平移不变性。因为照明，位置，比例等因图像而异。

下面来验证从零开始实现一下池化层的正向传播:

```python
import torch
from torch import nn

def pool2d(X, poole_size, mode='max'):
    p_h, p_w = poole_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2), 'max'), pool2d(X, (2, 2), 'avg')
```

#### 二维最大池化层
最常用的池化层。

**返回滑动窗口的最大值**:
![](https://s2.loli.net/2022/01/31/PsK3JcMODtBA5Tl.png)

我们常对卷积输出进行池化。

池化层的超参数：
* 池化层与卷积层类似,都具有填充和步幅.
* 没有可学习的参数
* 在每个输入通道应用池化层以获得相应的输出通道
* 输出通道数=输入通道数
*
对于多输入不会融合多个输入通道的，各自通道计算

#### 平均池化层
![](https://s2.loli.net/2022/02/01/M92BAI6znmJxqho.png)
*
最大池化层:每个窗口中最强的模式信号
* 平均池化层:将最大池化层中的“最大”操作替换为“平均”

下面我们使用**Pytorch**代码来进行实践:

#### 填充与步幅
需要注意的是深度学习框架中，**默认步幅和池化窗口大小相同**，当然可以手动设置。

```python
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
```

```python
# 深度学习框架中步幅和池化窗口大小相同
pool2d = nn.MaxPool2d(3)  # 3为3*3的窗口
pool2d(X)
```

手动指定

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

参数自定义

```python
# 设定一个任意大小的池化窗口，并分别设定填充和和步幅的高宽
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
```

通道上的单独计算

```python
X = torch.cat((X, X + 1), 1)  # 双通道
X
```

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

#### 总结一下
* 池化层返回窗口最大值或者平均值
* 作用是缓解卷积层位置的**敏感性**
* 同样有窗口大小，填充和步幅的超参数
