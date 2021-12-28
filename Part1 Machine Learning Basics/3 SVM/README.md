### 3 SVM

> 支持向量机
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics)

代码说明：

| 文件名  | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| svm     | sklearn工具包实现（使用高斯核函数）**由于训练速度较慢，建议手动选取部分数据进行训练即可** |
| svm_cmp | 对比选取不同核函数的可视化结果                               |

对比与logistic回归和神经网络，SVM在**非线性方程**很有优势！	

为了解释一些数学知识， 此文将用𝑧 表示$\theta^𝑇𝑥$。

#### 模型构建

* 引入

​	如果我们用一个新的代价函数来代替，即这条从 0 点开始的水平直线，然后是一条斜线，像上图。那么，现在让我给这两个方程命名，左边的函数，我称之为$cos𝑡_1(𝑧)$，同时，右边函数我称它为$cos𝑡_0(𝑧)$。这里的下标是指在代价函数中，对应的 𝑦 = 1 和 𝑦 = 0 的情况，拥有了这些定义后，现在，我们就开始构建支持向量机。

1. 回顾逻辑回归

$$
cost function = (-ylogh_\theta(x) + (1-y)log(1-h_\theta(x)))\\
其中，h_\theta(x) = \frac{1}{1 + e ^{-\theta^Tx}}
$$

2. 期望

   If $y = 1$ , $\theta^Tx >> 0 $ ;If $y=0$, $\theta^Tx <<0$

   

![](https://i.loli.net/2021/11/27/JYOzUM51Nug2SGB.jpg)

* 损失函数

$$
\min _{\theta} C \sum_{i=1}^{m}\left[y^{(i)} \operatorname{cost}_{1}\left(\theta^{T} x^{(i)}\right)+\left(1-y^{(i)}\right) \operatorname{cost}_{0}\left(\theta^{T} x^{(i)}\right)\right]+\frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$

当$𝜃^𝑇𝑥$大于或者等于 0 时，或者等于 0 时，所以学习参数𝜃就是支持向量机假设函数的形式。那么，这就是支持向量机数学上的定义。



* 加入正则项

$$
\min _{\theta} C \sum_{i=1}^{m}\left[y^{(i)} \cos t_{1}\left(\theta^{T} x^{(i)}\right)+\left(1-y^{(i)}\right) \operatorname{cost}_{0}\left(\theta^{T} x^{(i)}\right)\right]+\frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$

**支持向量机所做的是它来直接预测𝑦的值等于 1，还是等于 0。**



#### 深入探讨

* 大边界

  这是我的支持向量机模型的代价函数，在左边这里我画出了关于𝑧的代价函数$cos𝑡_1(𝑧)$，此函数用于正样本，而在右边这里我画出了关于𝑧的代价函数$cos𝑡_0(𝑧)$，横轴表示𝑧，现在让我们考虑一下，最小化这些代价函数的必要条件是什么。如果你有一个正样本，𝑦 = 1，则只有在𝑧 >= 1时，代价函数$cos𝑡_1(𝑧)$才等于 0。



​	具体而言，我接下来会考虑一个特例。我们将这个常数𝐶设置成一个非常大的值。**比如我们假设𝐶的值为 100000 或者其它非常大的数，**然后来观察支持向量机会给出什么结果？



​	**如果 𝐶非常大**，则最小化代价函数的时候，我们将会很希望找到一个使第一项为 0 的最优解。
$$
\min _{\theta} C \sum_{i=1}^{m}\left[y^{(i)} \cos t_{1}\left(\theta^{T} x^{(i)}\right)+\left(1-y^{(i)}\right) \cos t\left(\theta^{T} x^{(i)}\right)\right]+\frac{1}{2} \sum_{i=1}^{n} \theta_{j}^{2}
$$
![](https://i.loli.net/2021/11/27/8kjZMh7biIfcFRT.jpg)

​	黑线看起来是更稳健的决策界。这个距离叫做间距(**margin**)。

* 最小化代价函数

$\min \frac{1}{2} \sum_{j=1}^{n} \theta_{j}^{2}$ s.t 

$\left\{\begin{array}{c}\theta^{T} x^{(i)} \geq 1 \text { if } y^{(i)}=1 \\ \theta^{T} x^{(i)} \leq-1 i f y^{(i)}=0\end{array}\right.$

​	但很多时候一个异常值会影响你的决策边界，而如果正则化参数𝐶，设置的非常大，它将决策界，从黑线变到了粉线。

$C = 1 / \lambda$

* 𝐶 较大时，相当于 $\lambda$ 较小，可能会导致过拟合，**高方差**(过拟合)。
* 𝐶 较小时，相当于  $\lambda$  较大，可能会导致低拟合，**高偏差**（欠拟合）。



#### Kernel

> 核函数

* 假设我们的模型

$$
h_\theta(x) = \theta_{0}+\theta_{1} x_{1}+\theta_{2} x_{2}+\theta_{3} x_{1} x_{2}+\theta_{4} x_{1}^{2}+\theta_5x_2^2
$$

我们使用新的特征$f$来替换：
$$
f_1 = x_1, f_2 = x_2, f_3 = x_1x_2, f_4 = x_1^2, f_5 = x_2^2
$$

* 根据**近似程度**选取$f$

给定一个训练实例 𝑥 ，我们利用 𝑥 的各个特征与我们预先选定的**地标**(**landmarks**)$𝑙^{(1)}, 𝑙^{(2)}, 𝑙^{(3)}$的近似程度来选取新的特征$𝑓_1, 𝑓_2, 𝑓_3$
$$
f_1 = similarity(x, l^{(1)}) = e ^{\left(-\frac{\left\|x-l^{(1)}\right\|^{2}}{2 \sigma^{2}}\right)}\\
其中: \left\|x-l^{(1)}\right\|^{2}=\sum_{j=1}^{n}\left(x_{j}-l_{j}^{(1)}\right)^{2}
$$
$similarity(x, l^{(n)})$就是**核函数**，这里应该叫为**高斯核函数**。



* 直观解释

  * 当$x$和$l$的距离近似为0，这$e^{-0} = 1$
  * 当它们距离较远时候，$e^{大数} = 0$

  

* **细节补充**

> 一般情况下当训练数据集样本数为$m$的时候，我们地标选取的个数也是$m$

$$
f^{(i)}=\left[\begin{array}{c}
f_{0}^{(i)}=1 \\
f_{1}^{(i)}=\operatorname{sim}\left(x^{(i)}, l^{(1)}\right) \\
f_{2}^{(i)}=\operatorname{sim}\left(x^{(i)}, l^{(2)}\right) \\
f_{i}^{(i)}=\operatorname{sim}\left(x^{(i)}, l^{(i)}\right)=e^{0}=1 \\
\vdots \\
f_{m}^{(i)}=\operatorname{sim}\left(x^{(i)}, l^{(m)}\right)
\end{array}\right]
$$

运用到支持向量机当中：

* 当$\theta^Tf \geq 0$则预测$y=1$，反之为0

* 对应修改代价函数：$\sum_{j=1}^{n=m} \theta_{j}^{2}=\theta^{T} \theta$

* **为了简化运算**，我们将$\theta^TM\theta$代替$\theta^{T} \theta$
  * $M$是根据我们选择的核函数而不同的一个矩阵



在高斯核函数之外我们还有其他一些选择，如：

* 多项式核函数（**Polynomial Kerne**l）
* 字符串核函数（**String kernel**）
* 卡方核函数（ **chi-square kernel**）
* 直方图交集核函数（**histogram intersection kernel**）



#### 对比逻辑回归

> $m$为样本数量, $n$为特征数

1. 如果相较于$m$ 而言，$n$ 要大许多，即训练集数据量不够支持我们训练一个复杂的非线性模型，我们选用逻辑回归模型或者不带核函数的支持向量机。
2. 如果 $n$ 较小，而且 $m$ 大小中等，例如 $n$ 在 1-1000 之间，而 $m$ 在 10-10000 之间，使用带高斯核函数的支持向量机。
3. 如果 $n$ 较小，而 $m$ 较大，例如 $n$在 1-1000 之间，而 $m$ 大于 50000，则使用支持向量机会非常慢，解决方案是创造、增加更多的特征，然后使用逻辑回归或不带核函数的支持向量机。

   **值得一提的是，神经网络在以上三种情况下都可能会有较好的表现，但是训练神经网络可能非常慢**，选择支持向量机的原因主要在于它的代价函数是**凸函数**，不存在局部最小值。 