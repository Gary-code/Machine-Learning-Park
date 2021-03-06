### 9 EM算法（最大期望算法）

> 在前面[聚类的博客](https://blog.csdn.net/Garyboyboy/article/details/121865540)当中，我们简单的讲解过使用EM算法求解GMM模型的过程，这里我们对EM算法深入进行探讨。
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics)

> 代码说明：

| 文件名    | 说明                |
| --------- | ------------------- |
| gmm.ipynb | GMM模型的EM算法实现 |
| gmm.data  | 数据集文件          |

最大期望算法（Expectation-maximization algorithm），是在概率模型中**寻找参数最大似然估计**或者**最大后验估计**的算法，其中概率模型依赖于无法观测的隐性变量。

算法两个核心步骤：

* **E（计算期望）**
  * 利用对隐藏变量的现有估计值，计算其**最大似然估计值**
* **M（最大化）**
  * 最大化在E步上求得的最大似然值来计算参数的值
  * M步上找到的参数估计值被**用于下一个E步骤**计算中，这个过程不断交替进行。
* 简单的一句话表示就是：**知道结果，反推条件**$\theta$

#### 9.1 似然函数

**似然函数**是一种关于统计模型中的参数的函数，表示模型参数中的似然性。**极大似然就相当于最大可能。**

最大似然估计是已经知道了**结果**，然后寻求使该结果出现的可能性最大的**条件**，以此作为**估计值**。

**极大似然函数求解步骤**

> 我们通过一个例子来进行求解分析

假定我们要从10万个人当中抽取100个人来做身高统计，那么抽到这100个人的概率就是：

$L(\theta)=L\left(x_{1}, \ldots, x_{n} \mid \theta\right)=\prod_{i=1}^{n} p\left(x_{i} \mid \theta\right), \theta \in \Theta$

现在，我们的目标就是求解出$\theta$值，使得$L(\theta)$的值最大。

为此我们定义**对数似然函数**，将其变成连加的形式：

$H(\theta)=\ln L(\theta)=\ln \prod_{i=1}^{n} p\left(x_{i} \mid \theta\right)=\sum_{i=1}^{n} \ln p\left(x_{i} \mid \theta\right)$

在本科数学课程当中我们已经学过**偏导数**的求解方法，为此，我们对应求$L(\theta)$对所有参数的偏导数，也就是梯度了，从而$n$个未知的参数，就有$n$个方程，方程组的解就是似然函数的极值点了，最终得到这$n$个参数的值。

**极大似然函数估计值求解步骤如下**：

1. 写出似然函数；
2. 对似然函数取对数，并整理；
3. 求导数，令**导数为0**，得到似然方程；
4. 解似然方程，得到的参数即为所求；

#### 9.2 EM算法求解

> 同样，我们也通过一个例子来讲解EM算法

两枚硬币A和B，假定随机抛掷后正面朝上概率分别为$PA$，$PB$。为了估计这两个硬币朝上的概率，咱们轮流抛硬币A和B，每一轮都连续抛5次，总共5轮：

| 硬币 | 结果       | 统计    |
| ---- | ---------- | ------- |
| A    | 正正反正反 | 3正-2反 |
| B    | 反反正正反 | 2正-3反 |
| A    | 正反反反反 | 1正-4反 |
| B    | 正反反正正 | 3正-2反 |
| A    | 反正正反反 | 2正-3反 |

硬币A被抛了15次，在第一轮、第三轮、第五轮分别出现了3次正、1次正、2次正，所以很容易估计出PA，类似的，PB也很容易计算出来(**真实值**)，如下：

PA = （3+1+2）/ 15 = 0.4 PB= （2+3）/10 = 0.5

问题来了，如果我们**不知道抛的硬币是A还是B**呢（即硬币种类是**隐变量**），然后再轮流抛五轮，得到如下结果：

| 硬币    | 结果       | 统计    |
| ------- | ---------- | ------- |
| Unknown | 正正反正反 | 3正-2反 |
| Unknown | 反反正正反 | 2正-3反 |
| Unknown | 正反反反反 | 1正-4反 |
| Unknown | 正反反正正 | 3正-2反 |
| Unknown | 反正正反反 | 2正-3反 |

> 现在我们的目标没变，还是估计$PA$和$PB$，需要怎么做呢？

显然，此时我们多了一个硬币种类的隐变量，设为z，可以把它认为是一个**5维的向量**$（z1,z2,z3,z4,z5)$，代表每次投掷时所使用的硬币，比如$z1$，就代表第一轮投掷时使用的硬币是A还是B。

- 但是，这个变量$z$不知道，就无法去估计PA和PB，所以，我们必须先估计出$z$，然后才能进一步估计PA和PB。
- 可要估计z，我们又得知道PA和PB，这样我们才能用极大似然概率法则去估计$z$，这不是鸡生蛋和蛋生鸡的问题吗，如何解决呢？

解决方法：

* 先**随机初始化**一个$PA和PB$，用它来估计$z$
* 然后基于$z$，还是按照最大似然概率法则去估计新的$PA$和$PB$
* 然后依次循环，如果新估计出来的$PA和PB$和我们真实值差别很大，继续上一步过程，直到**PA和PB收敛到真实值为止。**

先随便给PA和PB赋一个值，比如： 硬币A正面朝上的概率$PA = 0.2$ 硬币B正面朝上的概率$PB = 0.7$

然后，我们看看第一轮抛掷最可能是哪个硬币。

 如果是**硬币A**，得出3正2反的概率为 :

$0.2 *0.2* 0.2 *0.8* 0.8 = 0.00512$ 

如果是**硬币B**，得出3正2反的概率为:

$0.7 *0.7* 0.7 *0.3* 0.3=0.03087$ 

然后依次求出其他4轮中的相应概率。做成表格如下：

| 轮数 | 若是硬币A        | 若是硬币B        |
| ---- | ---------------- | ---------------- |
| 1    | 0.00512，3正-2反 | 0.03087，3正-2反 |
| 2    | 0.02048，2正-3反 | 0.01323，2正-3反 |
| 3    | 0.08192，1正-4反 | 0.00567，1正-4反 |
| 4    | 0.00512，3正-2反 | 0.03087，3正-2反 |
| 5    | 0.02048，2正-3反 | 0.01323，2正-3反 |

按照最大似然法则： 第1轮中最有可能的是硬币B 第2轮中最有可能的是硬币A 第3轮中最有可能的是硬币A 第4轮中最有可能的是硬币B 第5轮中最有可能的是硬币A。

我们就把概率更大，即更可能是A的，即第2轮、第3轮、第5轮出现正的次数2、1、2相加，除以A被抛的总次数15（A抛了三轮，每轮5次），**作为$z$的估计值**，B的计算方法类似。然后我们便可以按照最大似然概率法则来估计新的PA和PB。
$$
PA = \frac{2+1+2}{15} = 0.33 \\
PB =\frac{3+3}{10} = 0.6
$$
就这样，不断迭代,不断接近真实值，这就是**EM算法**的神奇之处。

继续按照上面的思路，用估计出的$PA$和$PB$再来估计$z$，再用$z$来估计新的$PA$和$PB$，反复迭代下去，就可以最终得到$PA = 0.4$，$PB=0.5$，此时无论怎样迭代，$PA$和$PB$的值都会保持0.4和0.5不变，于是乎，我们就找到了$PA$和$PB$的最大似然估计。

**计算步骤总结**

1. 随机初始化分布参数$\theta$

2. E步，求$Q$函数，对于每一个$i$，计算根据上一次迭代的模型参数来计算出隐性变量的**后验概率**（**隐性变量的期望**），来作为隐藏变量的现估计值：
   $$
   Q_{i}\left(z^{(i)}\right)=p\left(z^{(i)} \mid x^{(i)} ; \theta\right)
   $$

3. M步，求使$Q$函数获得极大时的**参数取值**,将**似然函数最大化**以**获得新的参数值**。
   $$
   \theta=\operatorname{argmax} \sum_{i} \sum_{z^{(i)}} Q_{i}\left(z^{(i)}\right) \log \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}
   $$

4. 然后**循环重复**2、3步直到**收敛**。

用EM算法求解的模型一般有GMM或者协同过滤，k-means其实也属于EM。EM算法一定会收敛，但是**可能收敛到局部最优**。由于求和的项数将随着隐变量的数目**指数上升**，会给**梯度计算**带来麻烦。