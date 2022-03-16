## 2.3 训练技巧与实战
前面我们对神经网络的基本训练过程都进行了详细的介绍，但实践过程当中，我们通常会遇到很多情况导致我们的神经网络训练到某个程度之后就无法继续优化（前进），如何解决这个问题是2.3所要探讨的重点，我们主要分为：
* 局部最小值(local minima)和鞍点(saddle points)
* 批次(batch)与动量(momentum)
* 自动调整学习率(learning rate)  (*关于学习率的一些优化算法，会在后面的章节进行讲解与实践*)
* 损失函数选择(loss function) (*Part1已做过介绍这里不再讲解*)
* 批量标准化(batch normalization)

*实战部分*为**两个比赛**：
> 训练过程建议读者在[Colab](https://colab.research.google.com/)或者[Kaggle](https://www.kaggle.com/)上选用GPU加速跑模型。
* [COVID-19 Cases Prediction](https://www.kaggle.com/c/ml2021spring-hw1)
* [TIMIT framewise phoneme classification](https://www.kaggle.com/c/ml2021spring-hw2)

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

* [比赛2](./TIMIT framewise phoneme classification.ipynb)

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
