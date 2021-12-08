### 4 Ensemble Method

> 集成学习
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/4%20Ensemble%20Method](https://github.com/Gary-code/Machine-Learning-Park/tree/main/4%20Ensemble%20Method)

代码说明：

| 文件名                               | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| random_forest_example.ipynb          | 随机森林sklearn实现（使用iris数据集）                        |
| decision_tree_vs_random_forest.ipynb | 决策树与随机森林算法对比                                     |
| Adaboost文件夹                       | Adaboost用于人脸识别代码实现(lab文件夹下为手动实现代码，face_detection为opencv库自带方法) |
|                                      |                                                              |
|                                      |                                                              |

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

#### 4.3 AdaBoost（Adaptive Boosting）

​	在了解完决策树与随机森林过后，下面我们介绍AdaBoost模型。

与随机森林不一样，AdaBoost是Boosting算法中的一个显著代表。为此我们简单介绍一下Boosting算法。

##### Boosting

Boosting 算法的特征是个体学习器间存在强依赖关系，个体学习器以串行化方式生成。 

1. 首先依据初始训练集训练出一个基学习器，再根据基学习器的表现对**样本分布进行调整**，对基学习器做错的样本给予更高的权重, 即残差逼近的思想，以**减小偏差**。
2. 根据**调整后**的训练样本训练**下一个基学习器**，如此反复，直到达到预先指定的基学习器数目，再依据基学习器的表现进行结合，从而形成一个具有较好表现。

##### Adaboost

Adaboost的加权模型为**线性加权**，即：
$$
H(x)=\sum_{m=1}^{M} \alpha_{m} h_{m}(x)
$$


其伪代码如下：

![](https://s2.loli.net/2021/12/08/UxiTbVa7ljqk6Pe.png)

完成上面步骤之后，最后输出**模型**：
$$
H(\mathrm{x})=\operatorname{sign}\left(\sum_{t=1}^{T} \alpha_{t} h_{t}(\mathrm{x})\right)
$$
算法解释：

对第$t$个基学习器：

1. 选择并且拟合基学习器$h_t(x)$。
2. 根据拟合的基学习器，加入权重$w_{t}(i)$计算残差$e_t$。
3. 计算基学习器的权重$\alpha_t$。
4. 更新数据权重$w_{t+1}(i)$用于下一个基学习器。

仔细观察，我们可以把步骤4的计算过程改写成：
$$
w_{t+1}(i)= \begin{cases}\frac{w_{t}(i)}{z_{t}} e^{-\alpha_{t}}, & \text { for right predictive sample } \\ \frac{w_{t}(i)}{z_{t}} e^{\alpha_{t}}, & \text { for wrong predictive sample }\end{cases}
$$
同时，我们注意到，当$e_{t} \leq 0.5且\alpha_{t} \geq 0$的时候，$\alpha_{t}=\frac{1}{2} \ln \frac{1-e_{t}}{e_{t}}$随着$e_t$的减小,$\alpha_t$会变大。证明错误率较小的分类器将更为重要。也从中看出AdaBoost模型的合理性。

#### 4.4 GBDT（梯度提升决策树）

梯度提升决策树(GBDT)是一种具有迭代的决策树算法。

与AdaBoost相似，GBDT的关键是，树可以学习之前所有树的所有**结果**和**残差**。

梯度提升算法如下：

![](https://s2.loli.net/2021/12/08/Cx6ytu5hOTPaB1l.png)

#### 4.5 XGBoost

:rocket: ing...

#### 4.6 LGBM

:rocket:ing...