# 2 线性神经网络

> 注意：本页面的一些超链接跳转会有一些**错乱**，建议打开对应**小节**进行内容查看。

## 2.1 线性回归

> 在机器学习领域中的大多数任务通常都与预测有关。

那么就有一部分会涉及到回归问题（另外一部分会是分类问题，其目标是预测数据属于一组类别中的哪一个）。

* 例如：预测价格、课程参加人数预测。我们将会在[2.2](../2.2%20softmax回归/README.md)当中介绍多分类问题。

线性回归部分我们在[Part1](https://github.com/Gary-code/Machine-Learning-
Park/tree/main/Part1%20Machine%20Learning%20Basics/1%20LinearRegression)就有所介绍，也包含其[Pytorch的实现](https://github.com/Gary-
code/Machine-Learning-
Park/blob/main/Part1%20Machine%20Learning%20Basics/1%20LinearRegression/pytorch_version.ipynb)，这里我们忽略[Part1](https://github.com/Gary-
code/Machine-Learning-
Park/tree/main/Part1%20Machine%20Learning%20Basics/1%20LinearRegression)当中的一些理论知识，补充介绍一些相关知识，并且更加系统的完善一下实践的流程。

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import torch
```

### 2.1.1 矢量化加速

在训练我们的模型时，我们经常希望能够同时处理整个**小批量的样本**。 为了实现这一点，需要我们对计算进行矢量化，
从而利用**线性代数库**，而不是在Python中编写开销高昂的for循环。
下面我们通过一个实验说明。这里我们大量借鉴**d2l库**的源码来进行测试。首先实例化两个10000维的张量。

```python
n = 10000
a = torch.ones(n)
b = torch.ones(n)
```

为了方便起见，我们定义一个计时器Timer类。

```python
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
```

1. 首先我们使用for循环

```python
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'使用for循环{timer.stop():.5f} sec'
```

2. 使用重载过后的+运算

```python
timer.start()
d = a + b
f'矢量化加速下{timer.stop():.5f} sec'
```

通过结果可以看出来矢量化加速对张量运算十分重要。

### 2.1.2 正态分布与平方损失


随机变量$x$具有均值$\mu$和方差$\sigma^2$（标准差$\sigma$），其正态分布概率密度函数如下：
$p(x)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2
\sigma^{2}}(x-\mu)^{2}\right)$
下面使用Python定义正态分布的函数：

```python
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

可视化不同均值方差组合下的正态分布。

```python
x = np.arange(-10, 10, 0.01)

# 均值和方差组合
params = [(0, 1), (0, 2), (5, 1)]

for (mu, std) in params:
    plt.plot(normal(x, mu, std), label=f'mean {mu}, std {std}')

plt.legend()
plt.show()
```

如上图所示，改变均值会产生沿$x$轴的偏移，增加方差将会分散分布、降低其峰值(变得矮扁一些)。

均方误差损失函数（简称均方损失）可以用于线性回归的一个原因是： 我们假设了观测中包含噪声，其中噪声服从正态分布。 噪声正态分布如下式:
$$y=\mathbf{w}^{\top} \mathbf{x}+b+\epsilon$$
其中，$\epsilon \sim
\mathcal{N}\left(0, \sigma^{2}\right)$
因此，我们现在可以写出通过给定的$\mathbf{x}$观测到特定$y$的*似然*（likelihood）：

$$P(y \mid \mathbf{x}) =
\frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y -
\mathbf{w}^\top \mathbf{x} - b)^2\right).$$
现在，根据极大似然估计法，参数$\mathbf{w}$和$b$的最优值是使整个数据集的**似然**最大的值：

$$P(\mathbf y \mid
\mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$
根据极大似然估计法选择的估计量称为*极大似然估计量*。虽然使许多指数函数的乘积最大化看起来很困难，但是我们可以在不改变目标的前提下，通过最大化似然对数来简化。
**优化通常是说最小化**而不是最大化。

我们可以改为*最小化负对数似然*$-\log P(\mathbf y \mid \mathbf X)$。
由此可以得到：
$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2
\pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top
\mathbf{x}^{(i)} - b\right)^2.$$

*
现在只需要假设$\sigma$是某个固定常数就可以**忽略第一项**，因为第一项不依赖于$\mathbf{w}$和$b$。
*
现在第二项除了常数$\frac{1}{\sigma^2}$外，其余部分和前面介绍的均方误差是一样的。

* 幸运的是，上面式子的解并不依赖于$\sigma$。
  因此，在**高斯噪声的假设**下，**最小化均方误差等价于对线性模型的极大似然估计**。

**从线性回归 -> 深层神经网络**
尽管神经网络涵盖了更多更为丰富的模型，我们依然可以用描述神经网络的方式来描述线性模型。因此，和Part1最大的不同点在于，在这里会把线性模型看成一个神经网络，即便其为单层的神经网络，仍然具备我们未来所研究的网络的性质，我们使用下图对神经网络进行描述：
![单层神经网络](https://s2.loli.net/2022/01/14/Z3kRBH8yumcQPWo.png)

在上图所示的神经网络中:

* 输入为$x_1, \ldots, x_d$，因此输入层中的*输入数*（或称为*特征维度*，feature
  dimensionality）为$d$。
* 网络的输出为$o_1$，因此输出层中的*输出数*是1。
* 对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连，我们将这种变换（上图中的输出层）称为*全连接层*（**fully-connected layer**）或称为*稠密层*（dense layer)。

### 2.1.3 线性回归Pytorch实现

与[1.1中所展示代码](../../1%20AI框架使用(Pytorch)/1.3%20自动微分与简单训练实例/autodiff_training.ipynb)类似,熟悉流程的读者可以直接跳过。

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

6. 观察参数$w$和$b$的误差

```python
w = net[0].weight.data
print('w的估计误差为：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差为：', true_b - b)
```

## 2.2 Softmax回归

同样Softmax回归适用于解决多分类任务，常见的多分类任务有：

* 某个电子邮件是否属于垃圾邮件文件夹？
* 某个用户可能注册或不注册订阅服务？
* 某个图像描绘的是驴、狗、猫、还是鸡？
* 某人接下来最有可能看哪部电影？
  我们在[Part1](../../../Part1%20Machine%20Learning%20Basics/2%20LogisticRegression&Softmax/README.md)也有所介绍，这里我们对其理论部分不再做过多介绍。
  这里，我们重点对其理论体系进行一些扩展，并且实践Softmax回归的主要流程。

### 2.2.1 网络架构与损失函数

#### 网络架构

与2.1中所说的类似，同样对Softmax回归看成是一个单层神经网络的，其输出层同样为全连接层（FCN）。如下图所示，可以看到是多个输出（每个输出代表一个类别）：
![](https://s2.loli.net/2022/01/15/mDefkdzEqWwBpVs.png)

为了更简洁地表达模型，我们仍然使用线性代数符号。通过向量形式表达为$\mathbf{o} = \mathbf{W} \mathbf{x} +
\mathbf{b}$，这是一种更适合数学和编写代码的形式。

由此，我们已经将所有权重放到一个$3 \times
4$矩阵中。对于给定数据样本的特征$\mathbf{x}$，我们的输出是由权重与输入特征进行矩阵-向量乘法再加上偏置$\mathbf{b}$得到的。

与在[Part1](../../../Part1%20Machine%20Learning%20Basics/2%20LogisticRegression&Softmax/README.md)中讲解的公式一样，softmax函数将未规范化的预测变换为非负并且总和为1，同时要求模型保持可导。我们首先对每个未规范化的预测求幂，这样可以确保输出非负。为了确保最终输出的总和为1，我们再对每个求幂后的结果除以它们的总和。如下式：
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j
= \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$
这里，对于所有的$j$总有$0 \leq \hat{y}_j \leq
1$。因此，$\hat{\mathbf{y}}$可以视为一个正确的概率分布。softmax运算不会改变未规范化的预测$\mathbf{o}$之间的顺序，只会确定分配给每个类别的概率。因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别:
$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

#### 损失函数

接下来我们详细分析一下其损失函数的由来。这里我们采用极大似然估计的方式进行分析，与[2.1](../2.1%20线性回归/linear_regression.ipynb)的模型类似。

##### 对数似然

softmax函数给出了一个向量$\hat{\mathbf{y}}$，我们可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。

* 例如，$\hat{y}_1$=$P(y=\text{猫} \mid \mathbf{x})$
* 假设整个数据集$\{\mathbf{X},
  \mathbf{Y}\}$具有$n$个样本
* 其中索引$i$的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成

我们可以将估计值与实际值进行比较：
$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid
\mathbf{x}^{(i)}).
$$

根据最大似然估计，我们最大化$P(\mathbf{Y} \mid
\mathbf{X})$，相当于最小化负对数似然：

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n
-\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n
l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$
其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数为：
$$ l(\mathbf{y},
\hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$

##### softmax求导过程

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log
\frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log
\sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) -
\sum_{j=1}^q y_j o_j.
\end{aligned}
$$

考虑相对于任何未规范化的预测$o_j$的导数，得到：
$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q
\exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$
由此可见，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。从这个意义上讲，这与我们在回归中看到的非常相似，其中梯度是观测值$y$和估计值$\hat{y}$之间的差异。这不是巧合，在任何指数族分布模型中，对数似然的梯度正是由此得出的。这使梯度计算在实践中变得容易很多。
在训练softmax回归模型后，给出任何样本特征，我们可以预测每个输出类别的概率。通常我们使用预测概率最高的类别作为输出类别。

* 如果预测与实际类别（标签）一致，则预测是正确的。

* 在接下来的实践中，我们将使用*精度*（accuracy）来评估模型的性能。
* 精度等于正确预测数与预测总数之间的比值。

### 2.2.2 图像分类数据集构建

我们使用类似于MNIST但更复杂的**Fashion-MNIST数据集**，其主要不同点在包含了一些非数字的图片。

```python
%matplotlib inline
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
```

1. 读取数据集

* 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
* 并除以255使得所有像素的数值均在0到1之间

```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../../../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../../../data", train=False, transform=trans, download=True)
```

Fashion-MNIST由10个类别的图像组成， 每个类别由训练数据集（train dataset）中的6000张图像 和测试数据集（test dataset）中的1000张图像组成。 因此，训练集和测试集分别包含60000和10000张图像。 测试数据集不会用于训练，只用于评估模型性能。

```python
len(mnist_train), len(mnist_train)
```

每个输入图像的高度和宽度均为28像素。数据集由灰度图像组成，其通道数为1。为了简洁起见，本书将高度$h$像素、宽度$w$像素图像的形状记为$h \times
w$或（$h$,$w$）。

```python
mnist_train[0][0].shape
```

2. 可视化数据集

* Fashion-MNIST中包含的10个类别，分别为t-
  shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle
  boot（短靴）。
* 以下函数用于在数字标签索引及其文本名称之间进行转换。

```python
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

* 现在可以创建一个函数来可视化这些样本。

```python
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

### 2.2.3 Pytorch实现Softmax回归

> 为了演示方便我们使用d2l库

```python
# !pip install d2l  安装d2l库
```

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

1. 初始化模型参数
   *
   softmax回归的输出层是一个全连接层。因此，为了实现我们的模型，我们只需在`Sequential`中添加一个带有10个输出的全连接层。
   *
   我们仍然以均值0和标准差0.01随机初始化参数权重。

```python
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

#### 梯度消失（爆炸）

在前面的例子中，我们计算了模型的输出，然后将此输出送入交叉熵损失。从数学上讲，这是一件完全合理的事情。

然而，从计算角度来看，指数可能会造成数值稳定性问题。回想一下，softmax函数$\hat
y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat
y_j$是预测的概率分布。$o_j$是未规范化的预测$\mathbf{o}$的第$j$个元素。如果$o_k$中的一些数值非常大，那么$\exp(o_k)$可能大于数据类型容许的最大数字，即*上溢*（overflow）。这将使分母或分子变为`inf`（无穷大），最后得到的是0、`inf`或`nan`（不是数字）的$\hat
y_j$。在这些情况下，我们无法得到一个明确定义的交叉熵值。

解决这个问题的一个技巧是：在继续softmax计算之前，先从所有$o_k$中减去$\max(o_k)$。你可以看到每个$o_k$按常数进行的移动不会改变softmax的返回值：
$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j -
\max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& =
\frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$
在减法和规范化步骤之后，可能有些$o_j - \max(o_k)$具有较大的负值。由于精度受限，$\exp(o_j -
\max(o_k))$将有接近零的值，即*下溢*（underflow）。这些值可能会四舍五入为零，使$\hat y_j$为零，并且使得$\log(\hat
y_j)$的值为`-inf`。反向传播几步后，我们可能会发现自己面对一屏幕可怕的`nan`结果。

尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。

如下面的等式所示，我们避免计算$\exp(o_j\max(o_k))$，而可以直接使用$o_j - \max(o_k)$，因为$\log(\exp(\cdot))$被抵消了。
$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j -
\max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j -
\max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j -
\max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$
我们也希望保留传统的softmax函数，以备我们需要评估通过模型输出的概率。但是，我们没有将softmax概率传递到损失函数中，而是**在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数**，这是一种类似于[LogSumExp技巧](https://zhuanlan.zhihu.com/p/153535799)的方法。

2. 损失函数

```python
loss = nn.CrossEntropyLoss()
```

3. 优化算法

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

4. 训练

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## 2.3 训练技巧与实战

前面我们对神经网络的基本训练过程都进行了详细的介绍，但实践过程当中，我们通常会遇到很多情况导致我们的神经网络训练到某个程度之后就无法继续优化（前进），如何解决这个问题是2.3所要探讨的重点，我们主要分为：

* 局部最小值(local minima)和鞍点(saddle points)
* 批次(batch)与动量(momentum)
* 自动调整学习率(learning rate)  (*关于学习率的一些优化算法，会在后面的章节进行讲解与实践*)
* 损失函数选择(loss function) (*Part1已做过介绍这里不再讲解*)
* 批量标准化(batch normalization)

*实战部分*为**两个比赛**：

>训练过程建议读者在[Colab](https://colab.research.google.com/)或者[Kaggle](https://www.kaggle.com/)上选用GPU加速跑模型。

* [COVID-19 Cases Prediction](https://www.kaggle.com/c/ml2021spring-hw1)
* [TIMIT framewise phoneme classification](https://www.kaggle.com/c/ml2021spring-
  hw2)

### 2.3.1 训练技巧

在讲解训练中遇到的问题之前，我们现在对训练的*架构(Framework)*进行一下总览：

**Training**
Training data: $\left\{\left(x^{1}, \hat{y}^{1}\right),\left(x^{2},
\hat{y}^{2}\right), \ldots,\left(x^{N}, \hat{y}^{N}\right)\right\}$
训练步骤(Training Steps):

* Step 1: 初始化模型参数，$y=f_\theta(x)$
* Step 2:
  定义损失函数,$L(\theta)$
* Step 3: 优化，$\boldsymbol{\theta}^{*}=\arg \min
  _{\boldsymbol{\theta}} L$

**Testing**
Testing data: $\left\{x^{N+1}, x^{N+2},
\ldots, x^{N+M}\right\}$

* 使用$f_{\theta^*}(x)$预测测试集的标签。
  ![](https://s2.loli.net/2022/01/18/cX2tj8ULx6M3SwK.png)


**过拟合（Overfitting）**
如下图所示，训练集误差减小，但测试集误差很大，往往发生了过拟合的现象：
![](https://s2.loli.net/2022/01/18/5sMrJuzXgRk3xH9.png)

* 在数据层面上，解决方法就是训练*更多的数据*：
  ![](https://s2.loli.net/2022/01/18/FoBqPgenW7NkIRa.png)

* 而在模型层面上的解决方法有:
  * 更少的参数，或者共享参数(简化模型)
    ![](https://s2.loli.net/2022/01/18/JKjtgfakyQ2C54o.png)
  * 更少的特征
  * Early Stopping
  * 正则化(*Regularization*)
  * Dropout
  * 一个经典的例子就是CNN(*卷积神经网络*)
    ![](https://s2.loli.net/2022/01/18/v2f6GHNK49lXw5Y.png)

#### 更小的梯度

##### 局部最小值（Local minimal）与鞍点（Saddle point）

损失函数在*局部最小值*和*鞍点*的时候，梯度大小都会为0，但两者显著的区别如下图所示:
![](https://s2.loli.net/2022/01/18/JqjvaE7N9w841WP.png)
我们可以清楚的看到，鞍点的位置我们是有路可走的，但在局部最小值的地方我们会陷入一个“峡谷”当中。换而言之，鞍点情况下进行优化比在局部最小值继续优化更为*简单*。

为此我们需要借助数学的工具对这两种情况进行判定,可见[推理过程](http://www.offconvex.org/2016/03/22/saddlepoints/)

实际情况下，通过大量的实验证明，我们的模型会更多的处在鞍点的位置，而并非局部最小值处，因此训练过程中，我们完全可以大胆的进行梯度的调节。
![](https://s2.loli.net/2022/01/18/4aZnJARtYj3UsB7.png)

##### 批次（Batch）

在$\boldsymbol{\theta}^{*}=\arg \min _{\boldsymbol{\theta}}
L$过程当中，使用批次训练过程如下：
![](https://s2.loli.net/2022/01/18/gnLA5QKtkxHyerl.png)



实际上考虑到并行计算的因素，**大的批次**对训练时间是*没有显著的影响*（除非特别大的Batch Size），但**小的批次**运行完一个epoch需要花费*更长的时间*。
![](https://s2.loli.net/2022/01/18/TWlGrud1R4MUYAJ.png)
在*MNIST*和*CIFAR-10*的两个数据集当中，批次大小与准确度的关系如下所示：
![](https://s2.loli.net/2022/01/18/qX24w7KSTyBGQ1n.png)

所以Batch Size的合理设置十分重要，下面是关于一些Batch Size大小的对比：

| Batch Size           | Small      | Large                |
| -------------------- | ---------- | -------------------- |
| Speed for one update | Same       | Same (not too large) |
| Time for one epoch   | Slower     | **Faster**           |
| Gradient             | Noisy      | Stable               |
| Optimization         | **Better** | Worse                |
| Generalization       | **Better** | Worse                |

##### 动量(Momentum)

$m^t=\lambda m^{t-1} - \eta g^{t-1}$, $m^0=0$

使用动量前后对比：

* 前:
  ![](https://s2.loli.net/2022/01/18/aTc49M7XvFzASOo.png)
* 后
  ![](https://s2.loli.net/2022/01/18/eZUbiRzYgIxCo9s.png)
  在实际例子当中，动量可以让我们更容易跳出局部最小值，使得模型可以继续优化下去:
  ![](https://s2.loli.net/2022/01/18/vx4gGSVd9uYIrnC.png)

#### 批量标准化(*BN*)

仅仅对原始输入数据进行标准化是不充分的，因为虽然这种做法可以保证原始输入数据的质量，但它却无法保证隐藏层输入数据的质量。浅层参数的微弱变化经过多层线性变换与激活函数后被放大，改变了每一层的输入分布，造成深层的网络需要不断调整以适应这些分布变化，最终导致模型难以训练收敛。

简单的将每层得到的数据进行直接的标准化操作显然是不可行的，因为这样会破坏每层自身学到的数据特征。为了使“规范化”之后不破坏层结构本身学到的特征，BN引入了两个可以学习的“重构参数”以期望能够从规范化的数据中重构出层本身学到的特征。

* 计算批处理数据均值
  $$
  \mu_{B}=\frac{1}{m} \sum_{i=1}^{m} x_{i}
  $$

* 计算批处理数据方差
  $$
  \sigma_{B}^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{B}\right)^{2}
  $$
  

* 规范化
  $$
  \hat{x}_{i}=\frac{x_{i}-\mu_{B}}{\sqrt{\sigma_{B}^{2}+\epsilon}}
  $$
  

  使用BN后，可以:

* 缓解梯度消失，加速网络收敛。

* 简化调参，网络更稳定。BN层抑制了参数微小变化随网络加深而被放大的问题，对参数变化的适应能力更强，更容易调参。

* 防止过拟合。BN层将每一个batch的均值和方差引入到网络中，由于每个batch的这俩个值都不相同，可看做为训练过程增加了随机噪声，可以起到一定的正则效果，防止过拟合。

### 2.3.2 比赛

* [COVID-19 Cases
  Prediction](https://www.kaggle.com/c/ml2021spring-hw1)

* [TIMIT framewise
  phoneme classification](https://www.kaggle.com/c/ml2021spring-hw2)

  

  下面代码仅仅展示一个*最基础*的Baseline 代码：

* [比赛1](./covid19_prediction.ipynb)

* [比赛2](./phoneme_classification.ipynb)

并且将展示：

* 本地Windows环境（比赛1）下
  （*由于这个数据集较小才采用这种方式，否则建议使用kaggle或者colab来跑程序*）
* Colab环境（比赛2）下如何下载数据集

首先安装kaggle官方库

```python
!pip install kaggle
```

登录[Kaggle个人信息版块](https://www.kaggle.com/)，点击“Create New API
Token”下载kaagle.json文件:
![](https://s2.loli.net/2022/01/18/cF2IbDE7LKvUnkd.png)
随后新建'.kaggle'文件夹，将下载的json文件放入，并且整个文件夹移到C盘User目录下即可，最终如下所示:
![](https://s2.loli.net/2022/01/18/BHCn35UZ8sWEXAf.png)

**最后导入，kaggle包即可**
*注意*：用下面命令下载数据前，请务必先*同意该场比赛的规则*:
![](https://s2.loli.net/2022/01/18/HsbuUOEme6BK3o4.png)