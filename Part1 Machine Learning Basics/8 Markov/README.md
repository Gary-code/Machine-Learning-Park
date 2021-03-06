### 8 Markov（马尔科夫模型）

> HMM(隐马尔科夫模型)
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part1%20Machine%20Learning%20Basics)



> 代码说明：

| 文件名           | 说明                                     |
| ---------------- | ---------------------------------------- |
| markov_tag.ipynb | HMM模型对NLTK自带的Brown词库进行词性标注 |

隐马尔科夫模型（Hidden Markov Model，以下简称HMM）是比较经典的机器学习模型了，它在**自然语言处理**，**模式识别**等领域得到广泛的应用。

当然，随着深度学习的崛起，其逐渐被**RNN，LSTM，Transformer**等神经网络模型替代，但其中的思想仍然非常值得研究者去学习，由于本专栏专注于对基础知识的挖掘，因此对**HMM**会做深入的探究。

#### 8.1 模型基础

使用**HMM**的问题特征：

* 问题基于序列
  * 时间序列
  * 状态序列
* 两类数据
  * 可观测（观测序列）
  * 不可观测（隐藏序列）

> 比如：

​	我现在在打字写博客，我在键盘上敲出来的一系列字符就是**观测序列**，而我实际想写的一段话就是**隐藏序列**，输入法的任务就是从敲入的一系列字符尽可能的猜测我要写的一段话，**并把最可能的词语放在最前面让我选择**，这就可以看做一个HMM模型了。



> 下面我们通过数学形式来表达HMM模型

对于HMM模型，首先我们假设$Q$是所有可能的**隐藏状态的集合**，$V$是所有可能的**观测状态的集合**，即：
$$
Q=\left\{q_{1}, q_{2}, \ldots, q_{N}\right\}, V=\left\{v_{1}, v_{2}, \ldots v_{M}\right\}
$$
其中，$N$是可能的**隐藏状态数**，$M$是所有的可能的**观察状态数**。

对于一个长度为$T$的序列，$I$对应的状态序列, $O$是对应的观察序列，即：
$$
I=\left\{i_{1}, i_{2}, \ldots, i_{T}\right\}, O=\left\{o_{1}, o_{2}, \ldots o_{T}\right\}
$$
其中，任意一个隐藏状态$i_t \in Q$,任意一个观察状态$o_t \in V$



同时HMM模型做出两个重要的假设：

1. 齐次马尔科夫链假设。（**对隐藏状态而言**）即任意时刻的隐藏状态只依赖于它前一个隐藏状态，如果在时刻$t$的隐藏状态是$i_t=q_i$,在时刻$t+1$的隐藏状态是$i_{t+1}=q_{ji}$, 则从时刻tt到时刻$t+1$的HMM状态转移概率$a_ij$可以表示为：
   $$
   a_{i j}=P\left(i_{t+1}=q_{j} \mid i_{t}=q_{i}\right)
   $$
   $a_{ij}$可以组成==**马尔科夫链状态转移矩阵**$==$:
   $$
   A = [a_{ij}]_{N \times N}
   $$

2. 观测独立性假设。（**对于隐藏状态生成观测状态的概率**）即任意时刻的**观察状态**只**仅仅依赖于**当前时刻的**隐藏状态**，这也是一个为了简化模型的假设。

   如果在时刻$t$的隐藏状态是$i_t=q_j$, 而对应的观察状态为$o_t=v_k$, 则该时刻观察状态$v_k$在隐藏状态$q_j$下生成的概率为$b_j(k)$，则：
   $$
   b_{j}(k)=P\left(o_{t}=v_{k} \mid i_{t}=q_{j}\right)
   $$
   $b_j(k)$可以组成==**观测状态生成的概率矩阵**==：
   $$
   B=\left[b_{j}(k)\right]_{N \times M}
   $$

3. 除此意外，对于初始的状态信息，我们还要做出假设。要一组在时刻$t=1$的==**隐藏状态概率分布**==:
   $$
   \Pi=[\pi(i)]_{N} \text { 其中 } \pi(i)=P\left(i_{1}=q_{i}\right)
   $$
   
   
   

   一个HMM模型，可以由隐藏状态初始概率分布$\Pi$, 状态转移概率矩阵$A$和观测状态概率矩阵$B$决定。$\Pi,A$决定状态序列，$B$决定观测序列。因此，HMM模型可以由一个三元组$\lambda$表示如下：
   $$
   \lambda=(A, B, \Pi)
   $$
   

   #### 8.2 HMM例子

   > 例子来源于李航的《统计学习方法》

   假设我们有3个盒子，每个盒子里都有红色和白色两种球，这三个盒子里球的数量分别是：

   | 盒子   | 1    | 2    | 3    |
   | ------ | ---- | ---- | ---- |
   | 红球数 | 5    | 4    | 7    |
   | 白球数 | 5    | 6    | 3    |

​	按照下面的方法从盒子里抽球

* 开始的时候，从第一个盒子抽球的概率是0.2，从第二个盒子抽球的概率是0.4，从第三个盒子抽球的概率是0.4。以这个概率抽一次球后，将球放回。

* 然后从当前盒子转移到下一个盒子进行抽球。规则是：如果当前抽球的盒子是第一个盒子，则以0.5的概率仍然留在第一个盒子继续抽球，以0.2的概率去第二个盒子抽球，以0.3的概率去第三个盒子抽球。

  * 如果当前抽球的盒子是第二个盒子，则以0.5的概率仍然留在第二个盒子继续抽球，以0.3的概率去第一个盒子抽球，以0.2的概率去第三个盒子抽球。
  * 如果当前抽球的盒子是第三个盒子，则以0.5的概率仍然留在第三个盒子继续抽球，以0.2的概率去第一个盒子抽球，以0.3的概率去第二个盒子抽球。

* 如此下去，直到重复三次，得到一个球的颜色的观测序列:
  $$
  O = \{红,白,红\}
  $$

  >  注意在这个过程中，观察者**只能看到球的颜色序列**，却**不能看到球是从哪个盒子**里取出的。

  **观察集合**为$V=\{红，白\}，M=2$,**状态集合**为$Q=\{盒子1，盒子2，盒子3\}，N=3$

  观察序列和状态序列的长度为3

* 初始状态分布为：
  $$
  \Pi = (0.2, 0.4, 0.4)^T
  $$

* 状态转移概率分布矩阵$A$:
  $$
  A=\left(\begin{array}{lll}
  0.5 & 0.2 & 0.3 \\
  0.3 & 0.5 & 0.2 \\
  0.2 & 0.3 & 0.5
  \end{array}\right)
  $$

* 观测状态概率矩阵$B$:
  $$
  B=\left(\begin{array}{ll}
  0.5 & 0.5 \\
  0.4 & 0.6 \\
  0.7 & 0.3
  \end{array}\right)
  $$

#### 8.3 HMM模型特征

**HMM模型观察序列生成**

1. 根据初始状态概率分布$\Pi$生成隐藏状态$i_1$

2. for t from 1 to T
   1. 按照隐藏状态$i_t$的观测状态分布$b_{i_t}(k)$生成观察状态$o_t$
   2.  按照隐藏状态$i_t$的状态转移概率分布产生隐藏状态$i_{t+1}$
3. 所有的$o_t$一起形成观测序列$O=\{o_1,o_2,...o_T\}$



**经典的三个问题**

1） [评估观察序列概率](https://www.cnblogs.com/pinard/p/6955871.html)。需要用到**前向后向算法**。（简单）

2）[模型参数学习问题](https://www.cnblogs.com/pinard/p/6972299.html)。需要用到基于**EM算法**的鲍姆-韦尔奇算法。（难）

3）[预测问题](https://www.cnblogs.com/pinard/p/6991852.html)，也称为解码问题。需要用到基于**动态规划的维特比算法**。（居中）



