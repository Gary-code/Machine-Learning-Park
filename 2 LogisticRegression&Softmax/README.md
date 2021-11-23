### 2 Logistic Regression & Softmax

#### 2.1 Logistic Regression

> 逻辑回归
>
> 本文Github仓库已经同步文章与代码[https://github.com/Gary-code/Machine-Learning-Park/tree/main/2%20LogisticRegression%26Softmax](https://github.com/Gary-code/Machine-Learning-Park/tree/main/2%20LogisticRegression%26Softmax)
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