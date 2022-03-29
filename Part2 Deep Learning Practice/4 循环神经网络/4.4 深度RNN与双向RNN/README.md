## 4.4 深度循环神经网络与双向循环神经网络

```python
import torch
from torch import nn
from d2l import torch as d2l

# 加载数据集，祖传时间机器数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### 4.4.1 深度RNN

我们可以将多层循环神经网络堆叠在一起， 通过对几个简单层的组合来新成深度循环神经网络。其模型架构图如下所示:

![image-20220209101701578](https://s2.loli.net/2022/02/09/N87FWXUhrwPRbBk.png)

$$
\begin{aligned}
&\mathbf{H}_{t}^{1}=f_{1}\left(\mathbf{H}_{t-1}^{1}, \mathbf{X}_{t}\right) \\
&\quad \ldots \\
&\mathbf{H}_{t}^{j}=f_{j}\left(\mathbf{H}_{t-1}^{j}, \mathbf{H}_{t}^{j-1}\right) \\
&\quad \cdots \\
&\quad \mathbf{O}_{t}=g\left(\mathbf{H}_{t}^{L}\right)
\end{aligned}
$$

下面我们使用`pytorch`进行简洁实现：

与之前小节唯一的区别是，现在通过`num_layers`的值来设定隐藏层数。

```python
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

由于本人资源有限，为了避免花费过长时间训练，将训练代码注释掉。读者若想观看训练效果，可以将`#`去掉，然后执行代码查看。

```python
num_epochs, lr = 500, 2
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

### 4.4.2 双向（Bi-direction）RNN

在之前讲解的RNN当中，我们都是从头往前看得，但是有时候未来的信息很重要，比如你在做英语试卷的完型填空的时候，你往往需要结合**上下文**才能选出正确的答案。不过需要注意的是，并不是所有任务都可以使用双向的神经网络的。

![image-20220209144031976](https://s2.loli.net/2022/02/09/qLOUlfkao4evMTr.png)

* 取决于过去和未来的上下文，可以填很不一样的词
* 目前为止RNN只看过去
* 在**完型填空**的时候,我们也可以看未来


双向RNN的结构如下所示:
![image-20220209144346715](https://s2.loli.net/2022/02/09/8HNcjw2FLsDPCtY.png)

可以看到:
* 一个**前向RNN隐层**。
* 一个**后向RNN隐层**。
* **合并**两个隐状态得到输出。

其函数依赖可以表示为:
$$
\begin{aligned}
&\overrightarrow{\mathbf{H}}_{t}=\phi\left(\mathbf{X}_{t} \mathbf{W}_{x h}^{(f)}+\overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{h h}^{(f)}+\mathbf{b}_{h}^{(f)}\right) \\
&\overleftarrow{\mathbf{H}}_{t}=\phi\left(\mathbf{X}_{t} \mathbf{W}_{x h}^{(b)}+\overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{h h}^{(b)}+\mathbf{b}_{h}^{(b)}\right) \\
&\mathbf{H}_{t}=\left[\overrightarrow{\mathbf{H}}_{t}, \overleftarrow{\mathbf{H}}_{t}\right] \\
&\mathbf{O}_{t}=\mathbf{H}_{t} \mathbf{W}_{h q}+\mathbf{b}_{q}
\end{aligned}
$$



**关于双向RNN的注意事项:**

* 训练

![image-20220209144930318](https://s2.loli.net/2022/02/09/cxNvWEnzrkVqQRm.png)

* 推理
  * 非常**不适合做推理**

主要作用是对语义的特征提取，看齐整个句子，比如完型填空。

总结：
* 双向循环神经网络通过反向更新的隐藏层来利用方向时间信息
* 通常用来对序列抽取特征、填空，而不是预测未来

`Pytorch`代码实践:

**注意**：下面双向循环神经网络的错误应用, 预测未来不能这样子的！！！（不能未卜先知）

```python
# 加载数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
# d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```
