# Mechine Learning Park

> 机器学习
>
> Gary 哥哥的哥哥
>
> 2021/11/15

## 介绍

> 本专栏主要分**两大版块**，主要用于分享与总结机器学习相关
>
> * **基础知识**
> * **当下应用**

* 每个核心知识点都包含**知识讲解**和代码实现。
* 本专栏主要目的用于个人学习，亦不可以篇概全，请谅解，有问题可提出。
* 主要代码实现使用python

先来介绍四类角色：

* 领域专家（Domain experts）：知晓机器学习的项目应该部署在那个领域当值，类似于产品经理。
* 数据科学家：类似一个全栈工程师。
* ML专家：制作和优化模型。
* 软件开发工程师（SDE）：产品的落地，需要维护很多其他管理资源的代码。（运维等）

借用斯坦福[实用机器学习课程](https://c.d2l.ai/stanford-cs329p/syllabus.html#ml-model-recap-i)的PPT来看一下这四类角色的成长：

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/image-1.png)





## 基础理论

> 本专栏主要介绍机器学习基础模型并给出对应python代码的实现

### 1 Linear Regression

> 线性回归

#### 1.1 Linear Algebra

> 线性代数

##### 矩阵与向量

$$
X = \left[\begin{array}{cc}
1402 & 191 \\
1371 & 821 \\
949 & 1437 \\
147 & 1448
\end{array}\right]
$$



> 一般默认都以列向量为研究对象
>
> * 为 $n \times 1$的矩阵

$$
y=\left[\begin{array}{l}
460 \\
232 \\
315 \\
178
\end{array}\right]
$$



##### 运算

* 矩阵乘法

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/20211119111147.png)

* 矩阵乘法的特征
  * 不满足交换律
  * 满足结合律

  $$
  A \times B \neq B \times A  \\
  A \times B \times C = A \times (B \times C)
  $$

  

* 单位矩阵： Identity Matrix

$$
I_{4 \times 4} = \left[\begin{array}{llll}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{array}\right]
$$


$$
AA^{-1} = A{-1}A = I \\
AI = IA = A
$$

##### 逆与转置

* 矩阵的逆：
  * 如矩阵𝐴是一个𝑚 × 𝑚矩阵（方阵），如果有逆矩阵，则：$ 𝐴𝐴^{-1}$ = $𝐴^{−1}𝐴$ = 𝐼

* 矩阵的转置：

  * $$
    \left|\begin{array}{ll}
    a & b \\
    c & d \\
    e & f
    \end{array}\right|^{T}=\left|\begin{array}{lll}
    a & c & e \\
    b & d & f
    \end{array}\right|
    $$

  * 基本性质
    $$
    \begin{aligned}
    &(A \pm B)^{T}=A^{T} \pm B^{T} \\
    &(A \times B)^{T}=B^{T} \times A^{T} \\
    &\left(A^{T}\right)^{T}=A \\
    &(K A)^{T}=K A^{T}
    \end{aligned}
    $$
    

#### 1.2 线性回归模型

##### 表达式

$$
Y = WX + b
$$

$w$为变量$X$的系数，$b$为偏置项。

##### 损失函数

> Loss Function -- MSE

$$
J =\frac{1}{2m}\sum_{i=1}^{m}(y^i - y)^2
$$

$m$为样本的个数

##### 模型

Hypothesis:      $\quad h(X)=b+WX$

Parameters:         $W, b$

Cost Function: $\quad J=\frac{1}{2 m} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2}$

Goal:            	$\quad \operatorname{minimize}_{b, W} J$



##### 求解

* 梯度下降算法

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/image-20210714205526599.png)

在梯度下降算法中，还有一个更微妙的问题，梯度下降中，我们要更新$𝜃_0$和$𝜃_1$ ，当 𝑗 = 0 和𝑗 = 1时，会产生更新，所以你将更新$𝐽(𝜃_0)$和$𝐽(𝜃_1)$。实现梯度下降算法的微妙之处是，在这个表达式中，如果你要更新这个等式，你需要同时更新$𝜃_0$和$𝜃_1$，我的意思是在这个等式中，我们要这样更新：$𝜃_0:= 𝜃_0$ ，并更新$𝜃_1:= 𝜃_1$。实现方法是：你应该计算公式右边的部分，通过那一部分计算出$𝜃_0$和$𝜃_1$的值，然后同时更新$𝜃_0$和$𝜃_1$。

* $\alpha$是学习率，用来控制下降你要迈出多大的步子
* 完成一轮之后，更新两个系数$𝜃_0:= 𝜃_0$ ，$𝜃_1:= 𝜃_1$
  * **记住做完一轮才更新，不要一个一个系数轮着来更新**

* 如果𝑎太小了，即我的学习速率太小，结果就是只能这样像小宝宝一样一点点地挪动，去努力接近最低点，这样就需要很多步才能到达最低点，所以如果𝑎太小的话，可能会很慢，因为它会一点点挪动，它会需要很多步才能到达全局最低点。

* 如果𝑎太大，那么梯度下降法可能会越过最低点，甚至可能无法收敛，下一次迭代又移动了一大步，越过一次，又越过一次，一次次越过最低点，直到你发现实际上离最低点越来

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/image-20210714211745067.png)

* 批量梯度下降
  * 指的是在梯度下降的每一步中，它是指在**每一次迭代时**使用**所有样本**来进行梯度的更新。

$$
\theta_j := \theta_j - \alpha  \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}
$$

对比随机梯度下降

* **每次迭代**使用**一个样本**来对参数进行更新。使得训练速度加快。
* $repeat${
      for $i=1,...,m${

$$
\theta_j := \theta_j -\alpha (h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}  (for j =0,1)
$$

​		}
 }

##### 进阶

* 尺度缩放

在我们面对多维特征问题的时候，**我们要保证这些特征都具有相近的尺度**（范围类似），这将帮助梯度下降算法更快地收敛。
$$
X_𝑛 = \frac{X_n-\mu_n}{s_n}
$$
其中 $𝜇_𝑛$是平均值，$𝑠_𝑛$是标准差(即$max - min$)。



* 学习率 $\alpha$
  * 通常可以考虑尝试些学习率：$\alpha = 0.01, 0.03 ,0.1, 0.3 ,1, 3, 10$



* 正规方程

> normal Equation

$$
𝜃 = (𝑋^𝑇𝑋)^{-1}𝑋^𝑇𝑦
$$

使用python实现如下：

```python
import numpy as np
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y #X.T@X 等价于 X.T.dot(X)
    return theta
```

**注意**！！：**对于那些不可逆的矩阵**，正规方程方法是不能用的。

在大规模的数据时候**优先使用梯度下降**而非正规方程



* 正则项

$$
J = J_{原} + \lambda \sum_{j=1}^{m}\theta_j^2\\
J_{原} = \frac{1}{2 m} \sum_{i=1}^{m}\left(h\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$

为了防止过拟合。



### 2 Logistic Regression & Softmax

#### 2.1 Logistic Regression

> 逻辑回归
>
> * **这是解决分类问题而不是回归问题的！**

> 代码文件说明

| 文件名                        | 说明                          |
| ----------------------------- | ----------------------------- |
| logistic_numpy.ipynb          | 逻辑回归实现（使用a9a数据集） |
| softmax_pytorch_version.ipynb | softmax实现                   |



##### 模型构建

* 模型假设

模型的假设是： $ℎ_𝜃(𝑥) = 𝑔(𝜃^𝑇𝑋) $其中： 𝑋 代表特征向量 𝑔 代表逻辑函数（**logistic** **function**)是一个常用的逻辑函数为 **S** 形函数（**Sigmoid function**），公式为：$ 𝑔(𝑧) = \frac{1}{1+𝑒^{−𝑧} }$

```python
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

* 函数图像

  ![image](https://camo.githubusercontent.com/f3c61b86afee88892afd380ea1f172a358f639cb825d505c506584107384f2a4/68747470733a2f2f7778342e73696e61696d672e636e2f6c617267652f30303633304465666c7931673470766b32637461746a333063773062363379712e6a7067)

* 举个例子：

如果对于给定的𝑥，通过已经确定的参数计算得出$ℎ_{\theta}(𝑥) = 0.7$，则表示有 70%的几率𝑦为正向类，相应地𝑦为负向类的**几率为 1-0.7=0.3**

* 判断边界

当$ℎ_𝜃(𝑥) $>= 0.5时，预测 𝑦 = 1。 

当$ℎ_𝜃(𝑥) $< 0.5时，预测 𝑦 = 0 。

又 $𝑧 = 𝜃^𝑇𝑥$ ，即：

$𝜃^𝑇𝑥$ >= 0 时，预测 𝑦 = 1 

$𝜃^𝑇𝑥$ < 0 时，预测 𝑦 = 0

* 损失函数

对于线性回归模型，我们定义的损失函数是所有模型误差的平方和。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将$ℎ_𝜃(𝑥) = \frac{1}{1+𝑒^{1−𝜃^𝑇𝑋}}$ 

代入到这样定义了的损失函数中时，**我们得到的损失函数将是一个非凸函数（non-convexfunction）。**

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/99d365930166c7a1c19da7e6c818a13.jpg)

这意味着我们的损失函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。

因此我们需要重新定义损失函数：
$$
\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}
-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{aligned}\right. \\
$$

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}Cost(h_\theta(x), y)
$$

```python
import numpy as np
def cost(theta, X, y):
    theta = np.matrix(theta)
 	X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
    return np.sum(first - second) / (len(X))
```

> 下面我们对损失函数进行优化

合并上面的式子可以得到：
$$
𝐶𝑜𝑠𝑡(ℎ_𝜃(𝑥), 𝑦) = −𝑦 × 𝑙𝑜𝑔(ℎ_𝜃(𝑥)) − (1 − 𝑦) × 𝑙𝑜𝑔(1 − ℎ_𝜃(𝑥))
$$


* 梯度下降算法求解



> 当然除了**SGD**（梯度下降算法）之外，还有其他的求解方法，但这里不做过多介绍。

![](https://raw.githubusercontent.com/Gary-code/Machine-Learning-Park/files/blogImgs/image-20210721000458630.png)

#### 2.2 softmax与多分类问题

> 实际上Softmax回归也是一种**分类算法**，对比**逻辑回归**：
>
> * 从二元分类变成多输出分类

##### 模型构建

* 有效编码

> 这里我们以独热编码（one-hot coding）来举例

$$
\begin{aligned}
&\mathbf{y}=\left[y_{1}, y_{2}, \ldots, y_{n}\right]^{\top} \\
&y_{i}=\left\{\begin{array}{l}
1 \text { if } i=y \\
0 \text { otherwise }
\end{array}\right.
\end{aligned}
$$

* 无校验比例

最大值最为预测结果：
$$
\hat{y}=\underset{i}{\operatorname{argmax}} o_{i}
$$

* 校验比例（softmax）

  * 输出匹配概率

    * 非负
    * 和为1

    $$
    \begin{aligned}
    &\hat{\mathbf{y}}=\operatorname{softmax}(\mathbf{o}) \\
    &\hat{y}_{i}=\frac{\exp \left(o_{i}\right)}{\sum_{k} \exp \left(o_{k}\right)}
    \end{aligned}
    $$

* 损失函数

  > 交叉熵

  $$
  l(\mathbf{y}, \hat{\mathbf{y}})=-\sum y_{i} \log \hat{y}_{i}=-\log \hat{y}_{y}
  $$

  其梯度就是真实概率和预测概率的区别
  $$
  \partial_{o_{i}} l(\mathbf{y}, \hat{\mathbf{y}})=\operatorname{softmax}(\mathbf{o})_{i}-y_{i}
  $$

* 总结

  * Softmax回归是一 个多类分类模型
  * 使用Softmax操作子得到每个类的预测置信度
  * 使用交叉熵来来衡量预测和标号的区别



### 3 SVM

> 支持向量机
>
> * 对比与logistic回归和神经网络，SVM在非线性方程很有优势！

​	为了解释一些数学知识， 后续博客将用𝑧 表示$\theta^𝑇𝑥$。

#### 模型构建

* 引入

​	如果我们用一个新的代价函数来代替，即这条从 0 点开始的水平直线，然后是一条斜线，像上图。那么，现在让我给这两个方程命名，左边的函数，我称之为$cos𝑡_1(𝑧)$，同时，右边函数我称它为$cos𝑡_0(𝑧)$。这里的下标是指在代价函数中，对应的 𝑦 = 1 和 𝑦 = 0 的情况，拥有了这些定义后，现在，我们就开始构建支持向量机。

![](https://i.loli.net/2021/11/24/rDki8h7n2BRyfFJ.png)

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
f_1 = x_1, f_2 = x_2, f3 = x_1x_2, f4 = x_1^2, f5 = x_2^2
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

### 4 Ensemble Method

> 集成学习

代码说明：

| 文件名              | 说明                                  |
| ------------------- | ------------------------------------- |
| random_forest.ipynb | 随机森林sklearn实现（使用iris数据集） |
|                     |                                       |
|                     |                                       |
|                     |                                       |
|                     |                                       |

什么是集成学习？

* 结合几个基本模型，生成一个更好的预测模型
* 主要方法：Bagging、Boosting

![](https://i.loli.net/2021/11/30/mUQdz3GVhDZP5xu.jpg)

* Bootstraping：有放回采样

* Bagging：有放回采样$n$个样本

  通过bootstrap采样获得$T$个采样集

  分别通过样本集训练$T$个基础学习器

  * 对于分类问题：选票最多的类别成为最后一个类别。
  * 对于回归问题：最终的输出是每个基础学习器的平均输出。

  

本章节我们将围绕6种集成学习模型算法来进行讲解：

* 决策树
* AdaBoost
* 随机森林
* GBDT
* XGBoost
* LGBM



#### 4.1 决策树

> 决策树是一种有监督的机器学习算法，可用于**分类**和**回归**问题。

通过一个银行决定是否给用户贷款例子来看一下决策树是如何运作的：

![img](https://img2020.cnblogs.com/other/1981858/202006/1981858-20200618202354959-1195967773.jpg)

因此，决策树根据数据中的一组特征属性（在本例中为信用历史（**Credit History**）、收入（**Income**）和贷款金额（**Loan Amount**））做出一系列决策。

##### 决策树生成算法

> 算法伪代码如下：

![](https://s2.loli.net/2021/12/05/Ox8hKjZBt42kqmY.png)

​	但在很多实际问题当中，数据的某些属性值是对决策**并不重要**或者**毫不相关**。为此我们要对**特征的好坏**进行辨别即**属性选择**。

​	选择属性的算法有很多，其中我们可以使用以下指标来度量：

* 信息增益
* 增益率
* 基尼指数

> 给定带有正类和负类的数据集$D$
>
> * $p_+$为正类占总样本的比例
> * $p_-$为正类占总样本的比例

在讲解**属性选择**方法前，先了解**信息熵**的相关知识

##### **信息熵**（Entropy）

​	描述任意实例集合的**纯度**（purity）

​	**纯度越高**，$Entropy$**值越小**

* 对于二元分类问题

​	其公式为：
$$
Entropy(D) = -p_{+} \log _{2} p_{+}-p_{-} \log _{2} p_{-}
$$


​	如果值为0，则表示$D$中所有样本都属于同一个标签。（相同类别）

​	因为$(p_-) + (p_+) = 1$，所以$Entropy(D)-p_+$函数图像为：

![](https://s2.loli.net/2021/12/05/LwMIRmoTV6gqDt5.png)

* 对于多分类问题
  $$
  \operatorname{Entropy}(D) = \sum_{n=1}^{c}-p_{i} \log _{2} p_{i}
  $$
  $p_i$为第$i$类样本所占总样本的比例

##### **信息增益**（Information gain）

信息增益反映的是通过属性 $𝐴$ 划分训练集 $𝐷$ 能够带来的纯度提升量，记为$Gain(𝐷, 𝐴)$ 

* $Gain$值**越大**，意味着分后的纯度**提升越大**，属性 $𝐴$ 的划分**效果越好**。

其公式如下：
$$
\operatorname{Gain}(D, A)=\operatorname{Entropy}(D)-\sum_{j=1}^{v} \frac{\left|D_{j}\right|}{|D|} \operatorname{Entropy}\left(D_{j}\right)
$$
其中$D_j$是数据集$D$划分成$v$个的子集。

我们目标是希望挑选出具有尽可能好的划分效果的属性$A$ 。即我们希望属性$A$划分过后的子集应该尽可能的纯净，$Entroy(D_j)$越小越好。因此上面式子第二项就是$A$属性划分过后信息熵的**数学期望**。

实际上，式子等价于：
$$
\text { Entropy }(D)-\text { Entropy }_{A}\left(D_{j}\right)
$$


> 计算例子：

![img](https://s2.loli.net/2021/12/05/MGKlB4fVsdQP5wD.png)

将数据二值化：

![img](https://s2.loli.net/2021/12/05/Tjfni6zv1UHlN3V.png)

计算：

![](https://s2.loli.net/2021/12/05/1oOJBxXYRkLITDj.png)

##### 增益率

> 如果将类似**样本标识符**当成属性来计算增益率，会带来一些问题。

由于信息增益挑选属性总是趋于选择取值更大的属性，我们可以使用增益率来**减少**信息增益带来的一些影响。

公式如下：
$$
\operatorname{Gain\_ratio}(D, A)=\frac{\operatorname{Gain}(D, A)}{\operatorname{IV}(A)}
$$
其中
$$
I V(A)=-\sum_{j=1}^{v} \frac{\left|D_{j}\right|}{|D|} \log _{2} \frac{\left|D_{j}\right|}{|D|}
$$
$IV(A)$是属性$A$的**固有属性**，表示属性$A$的可能取值数量，数量越多，$IV(A)$的值越大。



但增益率可能会导致倾向于选择可取值更少的一些属性，因此我们一般处理步骤如下：

1. 先从$D$划分属性中找出**信息增益高于平均水平**的属性。
2. 然后从中选择**增益率高**的属性。



##### 基尼系数

* 基尼值

$$
Gini(D)=\sum_{k=1}^{K} p_{k}\left(1-p_{k}\right)=1-\sum_{k=1}^{K} p_{k}^{2}
$$

其中$D$有$K$个类，$p_k$为第$k$个类样本的频率。

$Gini(D)$的值**越高**，数据集$D$**越混乱**

其**基尼系数**表达式如下：
$$
\operatorname{Gini\_index}(D, A)=\sum_{j=1}^{v} \frac{\left|D_{j}\right|}{|D|} \operatorname{Gini}\left(D_{j}\right)
$$
基尼系数**越小**，数据集**越纯净**，划分效果**越好**。

因此，在实践中选择**基尼指数最小**的属性作为最优划分属性。







#### 4.2 随机森林

> 随机森林使用具有随机结构的决策树

随机森林是以决策树为基础学习器的bagging算法的扩展

* Bagging：

​	Bagging策略可以帮助我们产生不同的数据集。从样本集（假设样本集$m$个数据点）中重采样选出$n$个样本（有放回的采样，样本数据点个数仍然不变为$N$），对这$n$个样本建立分类器（ID3、C4.5、CART、SVM、LOGISTIC），重复以上两步$m$次，获得$m$个分类器，最后根据这$m$个分类器的投票结果，决定数据属于哪一类。



* 随机森林

随机森林在**bagging的基础上更进一步**：

1.  确定基决策树的数量 $N$。我们可以通过调整树的规模来调整随机森林的预测效果。
2.  ==**样本的随机**==：利用自助采样法（bootstrap sampling）我们可以将样本数据分成**训练集**和**袋外数据**。
3.  ==**特征的随机**==：从所有属性中随机选取$K$​个属性，选择**最佳分割属性**作为节点建立CART决策树。（**这里面也可以是其他类型的分类器，比如SVM、Logistics**）重复这一过程直到满足我们设定的停止条件，得到一棵基决策树。
4.  估计袋外误差，我们可以利用袋外数据对生成的决策树计算出袋外误差。
5.  **重复过程 2 到 4**，直到得到有$N$**棵树**的随机森林。



如下图所示：

![](https://s2.loli.net/2021/12/05/gvoL96ZFa1DMEQu.jpg)

* 简单例子解释:

  根据已有的训练集已经生成了对应的随机森林，随机森林如何利用某一个人的年龄（Age）、性别（Gender）、教育情况（Highest  Educational Qualification）、工作领域（Industry）以及住宅地（Residence）共5个**特征**来预测他的收入层次。

  **收入层次：**

  Level 1 : 小于 $40,000

  Level 2: $40,000 – 150,000

  Level 3: 大于 $150,000

随机森林中每一棵树都可以看做是一棵**CART**（分类回归树），这里假设森林中有5棵CART树，总特征个数$N$=5，这里取$m$=1

> 这里假设每个CART树对应一个不同的特征

要预测的某测试样本的信息如下：

| 特征                               | 值             |
| ---------------------------------- | -------------- |
| Age                                | 35 years old   |
| Gender                             | Male           |
| Highest Educational  Qualification | Diploma holder |
| Industry                           | Manufacturing  |
| Residence                          | Metro          |

根据上面五棵CART树的分类结果，我们可以针对这个人的信息建立收入层次的分布情况：

![](https://i.loli.net/2021/11/30/dWQAyLeIUSnEFqg.png)

因此我们可以得出结论：

​	这个人的收入层次70%是一等，大约24%为二等，6%为三等，所以最终认定该人属于一等收入层次。



**与决策树之间对比：**

 对于一个测试数据，**将它投入到随机森林中的不同决策树中，会得到不同的测试结果**。若问题是一个分类问题，则可以通过求众数来得出测试数据的类别；若问题是一个回归问题，则可以通过求平均值得出测试数据的值。

* 随机森林对比决策树，具有更强的分割能力。
* 解决决策树泛化能力弱的缺点。



#### 4.3 AdaBoost



#### 4.4 GBDT

#### 4.5 XGBoost

#### 4.6 LGBM





## 应用实践

> 介绍当下工业界下的机器学习应用
>
> * 中途会插入一部分Kaggle上的相关比赛

### 1 数据

> 数据是前提和基础，用于模型训练和检测

#### 1.1 数据获取

> example：

* MNIST：手写数字。

* ImageNet：Google引擎上获取下来的。

* AudioSet：YouTube上声音的数据集。

* SquAD: 从维基百科上获取的问题答案对应对。

  **主要获取数据方法：**

  * 手动制作
  * 爬取数据集

  **哪些网站可以获取数据集：**

* PaperWithCodes
* Kaggle
* Open Data on AWS
* Google Dataset Search

#### 1.2 网页数据爬取

> 对网页某个特定数据感兴趣

#### 1.3 代码实现

##### Numpy实现

