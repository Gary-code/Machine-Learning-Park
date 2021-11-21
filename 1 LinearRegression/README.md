### 1 Linear Regression

> 线性回归
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/1%20LinearRegression](https://github.com/Gary-code/Machine-Learning-Park/tree/main/1%20LinearRegression)

代码说明：

| 文件名          | 说明                                    |
| --------------- | --------------------------------------- |
| numpy_version   | numpy实现使用data.txt数据集             |
| pytorch_version | 包括使用pytorch从零开始实现和调用包实现 |





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

​	}
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

