# 6 优化算法和计算性能

## 6.1 优化算法理论

> 本小节理论公式较多，其中夹着一些代码演示方便读者理解。

### 6.1.1 优化和深度学习

尽管优化提供了一种最大限度地减少深度学习损失函数的方法，但实质上，优化和深度学习的目标是根本不同的：

* 前者主要关注的是**最小化目标**
* 后者则关注在给定有限数据量的情况下寻找**合适的模型**。

优化的目标是减少**训练误差**。但是，深度学习（或更广义地说，统计推断）的目标是**减少泛化误差**。为了实现后者，除了使用优化算法来减少训练误差之外，我们还需要注意过拟合。

下面我们通过实际的例子来展示：

```python
%matplotlib inline
import numpy as np
import torch
from mpl_toolkits import mplot3d
from d2l import torch as d2l
```

经验风险是训练数据集的**平均损失**，而风险则是整个数据群的预期损失。

下面我们定义了两个函数：风险函数`f`和经验风险函数`g`。假设我们只有有限量的训练数据。因此，这里的`g`不如`f`平滑。

```python
def f(x):
    return x * torch.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * torch.cos(5 * np.pi * x)
```

下图说明，训练数据集的最低经验风险可能与最低风险（泛化误差）不同。

```python
def annotate(text, xy, xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = torch.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

**回顾一下之前的鞍点**

鞍点（saddle point）是指函数的所有梯度都消失但既不是全局最小值也不是局部最小值的任何位置。如下图所示：

```python
x = torch.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

如下例所示，较高维度的鞍点甚至更加隐蔽。考虑这个$f=x^2-y^2$函数。它的鞍点为$(0, 0)$。这是关于$y$的最大值，也是关于$x$的最小值。

```python
x, y = torch.meshgrid(
    torch.linspace(-1.0, 1.0, 101), torch.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

### 6.2 优化算法

#### 6.2.1 动量法

> 由于之前的章节对此有所介绍这里不展开讨论，我们看看它的代码实现

$$
\begin{aligned}
&\mathbf{v}_t \leftarrow \beta \mathbf{v}_{t-1}+\mathbf{g}_{t, t-1} \\
&\mathbf{x}_t \leftarrow \mathbf{x}_{t-1}-\eta_t \mathbf{v}_t
\end{aligned}
$$

相比于小批量随机梯度下降，动量方法需要维护一组辅助变量，即速度。 它与梯度以及优化问题的变量具有相同的形状。 在下面的实现中，我们称这些变量为`states`。


* **从零开始实现**

```python
def init_momentum_states(feature_dim):
    v_w = torch.zeros((feature_dim, 1))
    v_b = torch.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```python
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

当我们将动量超参数momentum增加到0.9时，它相当于有效样本数量增加到$\frac{1}{1-0.9}=10$。我们将学习率略微降至0.01，以确保可控。

```python
train_momentum(0.01, 0.9)
```

降低学习率进一步解决了任何非平滑优化问题的困难，将其设置0.005为会产生良好的收敛性能。

```python
train_momentum(0.005, 0.9)
```

* 简洁实现

```python
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

#### 6.2.2 AdaGrad算法

假设我们正在训练一个语言模型。为了获得良好的准确性，我们大多希望在训练的过程中降低学习率，速度通常为$\mathcal{O}(t^{-\frac{1}{2}})$或更低。

现在讨论关于稀疏特征（即只在偶尔出现的特征）的模型训练，这对自然语言来说很常见。
例如，我们看到“预先条件”这个词比“学习”这个词的可能性要小得多。但是，它在计算广告学和个性化协同过滤等其他领域也很常见。

**只有在这些不常见的特征出现时，与其相关的参数才会得到有意义的更新。**
鉴于学习率下降，我们可能最终会面临这样的情况：

* 常见特征的参数相当迅速地收敛到最佳值
* 而对于不常见的特征，我们仍缺乏足够的观测以确定其最佳值。

换句话说，学习率要么对于**常见特征而言降低太慢**，要么对于**不常见特征而言降低太快**。解决此问题的一个方法是记录我们看到特定特征的次数，然后将其用作调整学习率。

我们使用变量$\mathbf{s}_t$来累加过去的梯度方差，如下所示：

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

在这里，操作是按照坐标顺序应用。也就是说，$\mathbf{v}^2$有条目$v_i^2$。
同样，$\frac{1}{\sqrt{v}}$有条目$\frac{1}{\sqrt{v_i}}$，并且$\mathbf{u} \cdot \mathbf{v}$有条目$u_i v_i$。

与之前一样，$\eta$是学习率，$\epsilon$是一个为维持数值稳定性而添加的常数，用来确保我们不会除以$0$。

最后，我们初始化$\mathbf{s}_0 = \mathbf{0}$。

就像在动量法中我们需要跟踪一个辅助变量一样，在AdaGrad算法中，我们允许每个坐标有单独的学习率。与SGD算法相比，这并没有明显增加AdaGrad的计算代价，因为主要计算用在$l(y_t, f(\mathbf{x}_t, \mathbf{w}))$及其导数。

请注意，在$\mathbf{s}_t$中累加平方梯度意味着$\mathbf{s}_t$基本上以线性速率增长（由于梯度从最初开始衰减，实际上比线性慢一些）。这产生了一个学习率$\mathcal{O}(t^{-\frac{1}{2}})$，但是在单个坐标的层面上进行了调整。对于凸问题，这完全足够了。

然而，在深度学习中，我们可能希望更慢地降低学习率。这引出了许多`AdaGrad`算法的变体，我们将在后续中讨论它们。眼下让我们先看看它在二次凸问题中的表现如何。我们仍然以同一函数为例：

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

我们将使用与之前相同的学习率来实现AdaGrad算法，即$\eta = 0.4$。
可以看到，自变量的迭代轨迹较平滑。但由于$\boldsymbol{s}_t$的累加效果使学习率不断衰减，自变量在迭代后期的移动幅度较小。

```python
import math
```

```python
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

我们将学习率提高到$2$，可以看到更好的表现。这已经表明，即使在**无噪声**的情况下，学习率的降低可能相当剧烈，我们需要确保参数能够适当地收敛。

```python
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

**从零开始实现`AdaGrad`算法**
同动量法一样，AdaGrad算法需要对每个自变量维护同它一样形状的状态变量。

```python
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

**简洁实现**
直接使用深度学习框架中提供的AdaGrad算法来训练模型。

```python
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

#### 6.2.3 Adam算法

Adam算法将所有这些技术汇总到一个高效的学习算法中。 不出预料，作为深度学习中使用的更**强大和有效**的优化算法之一，它非常受欢迎。


Adam算法的关键组成部分之一是：它使用指数加权移动平均值来估算梯度的动量和二次矩，即它使用状态变量

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

这里$\beta_1$和$\beta_2$是非负加权参数。
常将它们设置为$\beta_1 = 0.9$和$\beta_2 = 0.999$。
也就是说，方差估计的移动远远慢于动量估计的移动。
注意，如果我们初始化$\mathbf{v}_0 = \mathbf{s}_0 = 0$，就会获得一个相当大的初始偏差。
我们可以通过使用$\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$来解决这个问题。
相应地，标准化状态变量由下式获得

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

有了正确的估计，我们现在可以写出更新方程。
首先，我们以非常类似于RMSProp算法的方式重新缩放梯度以获得

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

与RMSProp不同，我们的更新使用动量$\hat{\mathbf{v}}_t$而不是梯度本身。
此外，由于使用$\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$而不是$\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$进行缩放，两者会略有差异。
前者在实践中效果略好一些，因此与RMSProp算法有所区分。
通常，我们选择$\epsilon = 10^{-6}$，这是为了在数值稳定性和逼真度之间取得良好的平衡。

最后，我们简单更新：

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

回顾Adam算法，它的设计灵感很清楚：
首先，动量和规模在状态变量中清晰可见，
它们相当独特的定义使我们移除偏项（这可以通过稍微不同的初始化和更新条件来修正）。
其次，RMSProp算法中两项的组合都非常简单。
最后，明确的学习率$\eta$使我们能够控制步长来解决收敛问题。

* 从零开始实现

从头开始实现Adam算法并不难。为方便起见，我们将时间步$t$存储在`hyperparams`字典中。除此之外，一切都很简单。

```python
def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

现在，我们用以上Adam算法来训练模型，这里我们使用$\eta = 0.01$的学习率。

```python
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

* 简洁实现

```python
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

### 6.3 预热

在某些情况下，初始化参数不足以得到良好的解。

这对于某些高级网络设计来说尤其棘手，可能导致不稳定的优化结果。 对此，一方面，我们可以选择一个足够小的学习率， 从而防止一开始发散，然而这样进展太缓慢。 另一方面，较高的学习率最初就会导致发散。


解决这种困境的一个相当简单的解决方法是**使用预热期**，在此期间学习率将增加至**初始最大值**，然后冷却直到优化过程结束。 为了简单起见，通常使用线性递增。 这引出了如下图片所示的时间表。

![](https://zh.d2l.ai/_images/output_lr-scheduler_1dfeb6_128_0.svg)

## 6.2 计算性能

本小节我们主要介绍与深度学习计算相关的**硬件与编译基础知识**，详细内容可见[d2l课程官网](https://zh.d2l.ai/chapter_computational-performance/hybridize.html), 若想学习机器学习编译(Machine Learning Compilation)相关知识，可以观看陈天奇的这门[课程](https://mlc.ai/summer22-zh/schedule), 和我本人对这门课程做的[简单笔记](https://github.com/Gary-code/MLC-Notes)

### 6.2.1 深度学习硬件: CPU 和 GPU

#### 6.2.1.1 CPU

下图为一张`i7 - 6700K`的结构图:
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211123201018.png)

为此我们通常会思考如何提升**CPU利用率**?
我们结合计算机组成原理中学到的知识来进行回答：
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211123201054.png)

答案就是利用CPU的**多核**, 下面进行一个简单样例的分析:
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124195623.png)
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124200011.png)

#### 6.2.1.2 GPU

下图是一张英伟达`Titan X`的示意图：
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124200131.png)

对比CPU：

* 多核和内存带宽高
* 但牺牲的就是内存不能做太大（太贵了）

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124201333.png)
![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124201547.png)

需要注意的是, 由于带宽限制和同步的开销，**不要在CPU和GPU之间来回跳转**。

### 6.2.1.3 其他计算处理器

* 高通845平台

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211124201645.png)

* DSP：数字信号处理
  ![](https://i.loli.net/2021/11/24/y1HcrflCM87PTua.png)

* 可编程阵列（FPGA）
  ![](https://i.loli.net/2021/11/24/FUCpV7diZGgnP89.png)

* AI ASIC
  ![](https://i.loli.net/2021/11/24/a6O8XEHZoYf1AN7.png)

### 6.2.2 单机多卡并行

* 小批量可以切割到不同GPU当中去并行加速
* 常用切分方案
  * 数据并行（主要的）
  * 模型并行
  * 通道并行

**数据并行**

* 将小批量分成n块，每个GPU拿到完整参数计算一块数据的梯度

* 通常性能更好

  ![](https://i.loli.net/2021/11/27/BaFmJitxjCzZP1q.png)

> 对数据来说并行

1. 读一个数据块
2. 拿回参数
3. 计算梯度
4. 发出梯度
5. 更新梯度

**模型并行**:

* 将模型分成n块，每个GPU拿到一块模型计算它的前向和方向结果
* 通常用于模型大到单GPU放不下


**总结**：

* 当一个模型可以在单卡计算时候，通常使用数据并行到多卡
* 模型并行主要用在超大型的模型上



### 6.2.3 分布式计算

#### 6.2.3.1 基础概念

**基本上等于单机多卡**

![](https://i.loli.net/2021/12/01/Kl9QiNgjdGM3wbz.png)

GPU机器架构

![](https://i.loli.net/2021/12/01/qVcm9RsSzb2Hh8T.png)



**计算一个小批量**

![](https://i.loli.net/2021/12/01/8nA9hwHeW6kotql.png)



同步SGD

* 这里每个worker都是同步计算一个批量 , 称为同步SGD
* 假设有n个GPU，每个GPU每次处理b个样本，那么同步SGD等价于在单GPU运行批量大小为nb的SGD
* **在理想情况下**，n个GPU可以得到相对个单GPU的n倍加速.

性能

* $t_1$=在单GPU上计算b个样本梯度时间
* 假设有m个参数，1个worker每次发送和接收m个参数、梯度
  * $t_2$=发送和接收所用时间
* 每个批量的计算时间为$max(t_1, t_2)$
  * 选取足够大的b使得$t_1>t_2$
  * **增加b或n**导致更大的批量大小，导致需要更多计算来得到给定的模型精度(**收敛变慢**)

![](https://i.loli.net/2021/12/01/HEpyleNJmkdcLsz.png)

#### 6.2.3.2 代码实践

* **使用一个大数据集**
* 需要好的GPU-GPU和机器机器**带宽**
* **高效的数据读取和预处理**
* 模型需要有好的计算(FLOP)通讯(model size)比
  * `Inception`> `ResNet` > `AlexNet`
* 使用足够大的批量大小来得到好的系统性能，
* 使用**高效的优化算法对对应大批量大小**

下面我们通过代码来实践一下多GPU计算(**由于本机无Nvidia的GPU，因此部分代码并未成功执行**)：

* 从零开始实现

```python
%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```python
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

```

```python
# 数据同步
def get_params(params, device):
    # 参数放到GPU上
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

```python
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

```python
# 将所有向量相加，并将结果广播给所有GPU
def allreduce(data):
    for i in range(1, len(data)):  # 先全部放到data[0]中，第0个GPU
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)
```

```python
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```python
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)  # 均匀切入对应GPU中
print('input :', data)
print('load into', devices)
print('output:', split)
```

```python
def split_batch(X, y, devices):
    """将`X`和`y`拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

```python
# 训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
            X_shards, y_shards, device_params)]
    for l in ls:  # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad
                       for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量
```

```python
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到`num_gpus`个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()  # 同步，用于算时间
        timer.stop()
        # 在GPU 0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```python
train(num_gpus=2, batch_size=256, lr=0.2)
```

* 简洁实现

`ResNet18`

```python
#@save
def resnet18(num_classes, in_channels=1):
    """稍加修改的 ResNet-18 模型。"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层。
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

```python
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 我们将在训练代码实现中初始化网络
```

```python
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多个 GPU 上设置模型
    net = nn.DataParallel(net, device_ids=devices)  # 切开放到多个GPU中
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```python
train(net, num_gpus=2, batch_size=256, lr=0.1)
```

#### 6.2.3.4 总结

* 分布式同步数据并行是多GPU数据并行在多机器上的拓展
* 网络通讯通常是瓶颈
* 需要注意使用**特别大的批量大小时收敛效率**
* 更复杂的分布式有异步、模型并行