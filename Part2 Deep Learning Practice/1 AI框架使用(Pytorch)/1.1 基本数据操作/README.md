# 1.1 基本数据操作
由于本人能力有限，不可能将所有Pytorch的操作都进行讲解。因此强烈建议读者遇到问题时候查阅Pytorch的[官方文档](https://pytorch.org/docs/stable/index.html)和参与一些论坛社区的讨论。

## 1.1.1 安装
对Pytorch的安装，这里也不做过多的展开介绍。可以来看[沐神的视频](https://www.bilibili.com/video/BV18p4y1h7Dr?spm_id_from=333.999.0.0)来进行学习。

## 1.1.2 张量与基本运算



为此我们首先导入torch

```python
import torch

# 为了后续方便我顺便将下面这些库也导入
import numpy as np
```

### Tensor的创建

可以通过我们熟悉的List或者Numpy来进行创建

```python
list_form = [[1, -1], [2, -2]]
x1 = torch.tensor(list_form)  # 从list中创建
x2 = torch.from_numpy(np.array(list_form))  # 从numpy中创建
x1, x2
```

当然tensor也可以转换为numpy

```python
x = x1.numpy()
x
```

其他类型tensor的创建
1. arange来进行创建

```python
x = torch.arange(12)
print(x)
x.shape, x.numel()  # 形状,数量
```

2. 空Tensor（size为$3 \times 4$）

```python
x = torch.empty(3, 4)
x
```

3. 随机初始化

```python
x = torch.rand(3, 4)  # 元素在(0, 1)之间
x
```

4. 单位tensor(元素全为1)

```python
x = torch.ones(3, 4)
x
```

5. 指定元素类型的tensor

```python
x = torch.ones(3, 4, dtype=torch.long)  # 指定long类型
x, x.dtype
```

6. 借助现有tensor创建tensor
此方法会默认重用输入Tensor的一些属性，如数据类型等

```python
x = torch.randn_like(x, dtype=torch.float)  # 正态分布，size与x一致
x
```

```python
x = x.new_ones(3, 4, dtype=torch.float)  # size为(3, 4)的单位tensor
x
```

### 基本运算操作

1. 简单四则运算，这里以加法为例

```python
x = x.new_ones(3, 4, dtype=torch.float)
y = torch.rand(3, 4)
x + y
```

```python
z = torch.add(x, y)
z
```

add_代表inplace版本。pytorch其他函数也类似如x.copy_(y), x.t_()

```python
y.add_(x)
y == z
```

2. 索引与形状

```python
y = x[0, :]
y += 1
y == x[0, :]  # 结果为True。证明源tensor也会改变
```

view和reshape是常用的改变tensor.shape的函数

```python
y = x.view(12)
z = x.view(-1, 6)  # -1所指的维度可以根据其他维度的值推出来
x.size(), y.size(), z.size()  # x.size开始时候为(3, 4)
```

深拷贝

```python
x += 1
x, y  # True, y的值也会跟着改变, 即使他们的shape不同。
```

因此如果我们想得到一个真正的副本而不是像上边那样共享内存，可以考虑使用reshape()函数。还有另外一个解决方案就是使用clone创建一个副本再使用view

```python
x_cp = x.clone().view(12)
x -= 1
x, x_cp  # x_cp不会跟着x改变
```

3. Squeeze/Unsqueeze去除(增加)长度为1的指定维度(具体更多的参数可以看官方文档)

* squeeze 去除
![](https://s2.loli.net/2022/01/10/ifH7OJDpz4CQndv.png)

```python
x = torch.zeros([1, 2, 3])
print(f'former shape:', x.shape)  # (1, 2, 3)
x.squeeze_(0)  # 可以指定维度，也可以不指定
print(f'shape after squeeze:', x.shape)
```

* 增加
![](https://s2.loli.net/2022/01/10/KeJY9iPyaCEu3WQ.png)

```python
x = torch.zeros([2, 3])
print(f'former shape:', x.shape)
x = x.unsqueeze(1)  # 在维度为1处添加
print(f'shape after unsqueeze:', x.shape)
```

4. 张量转置

```python
x = torch.zeros([2, 3])
x = x.transpose(0, 1)  # 转置的维度
x.shape
```

5. 连接多个tensor

```python
x = torch.zeros([2, 1, 3])
y = torch.zeros([2, 2, 3])
z = torch.zeros([2, 3, 3])
a = torch.cat([x, y, z], dim=1)  # 根据维度1来进行连接
a.shape  # (2, 6, 3)
```

## 1.1.3 广播机制(Broadcasting) 和内存问题

### 广播机制
即先适当复制元素使这两个Tensor形状相同后再按元素运算。

```python
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
y, x + y
```

### 内存问题
使用pytorch自带的id函数:
* 如果两个实例的ID一致，那么它们所对应的内存地址相同
* 反之则不同

```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
id(y) == id_before  # False
```

如果想指定结果到原来的y的内存，我们可以使用前面介绍的索引来进行替换操作。

我们把x + y的结果通过[:]写进y对应的内存中

```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x  # 仅仅改写元素
id(y) == id_before # True
```

还可以使用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果,如:
* ```torch.add(x, y, out=y)```
*
```y.add_(x)```
* ```y += x```

```python
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)  # 仅仅改写元素
id(y) == id_before # True
```

需要注意的是，虽然view返回的Tensor与源Tensor是**共享data的**，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者**id（内存地址）并不一致**。

### tensor的运算(利用广播机制)

1. 累计求和，特别注意axis参数

```python
a = torch.arange(20).reshape(5, 4)
print(f'a:', a)
b = a.sum(axis=0)
print(f'b:', b)
c = a.sum(axis=0, keepdim=True)  # 可以用广播机制，保留那个求和的维度
print(f'c:', c)
print(f'a/c:', a/c)

# 累加求和
print(f'a累加求和:', a.cumsum(axis=0))
```

2. 矩阵乘法
* 矩阵乘向量 ```mv```函数

```python
A = torch.rand(5, 4)
x = torch.rand(4)
A, x, A.shape, x.shape, torch.mv(A, x)
```

* 矩阵相乘  ```mm```函数

```python
B = torch.ones(4, 3)
torch.mm(A, B)
```

3. 范数
* $l2$范数
$\|x\|_{2}=\left(\left|x_{1}\right|^{2}+\left|x_{2}\right|^{2}+\left|x_{3}\right|^{2}+\cdots+\left|x_{n}\right|^{2}\right)^{1
/ 2}$

```python
u = torch.tensor([3., -4.])
torch.norm(u)
```

* $l1$范数
$||
x||_{1}=\left|x_{1}\right|+\left|x_{2}\right|+\left|x_{3}\right|+\cdots+\left|x_{n}\right|$

```python
u = torch.tensor([3., -4.])
torch.abs(u).sum()
```

* 矩阵$Frobenius$范数($F$范数，即元素平方和开根)
$\|X\|_{F} \stackrel{\text { def }}{=}
\sqrt{\sum_{i} \sum_{j} X_{i, j}^{2}}$

```python
torch.norm(torch.ones(4, 9))
```

## 1.1.4 其他操作

### 将tensor存放在GPU当中
首先，你需要确保你的Win/Linux机器拥有英伟达(NVIDIA)的显卡。[cuda的安装地址](https://developer.nvidia.cn/zh-
cn/cuda-toolkit)

在后面章节的模型训练中，不要频繁出现tensor在gpu和cpu之间跳转，否则训练时间会大大增加。

```python
if torch.cuda.is_available():  # 查看是否有cuda的设备
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()将tensor转移回去cpu,同时可以更改数据类型。
```
