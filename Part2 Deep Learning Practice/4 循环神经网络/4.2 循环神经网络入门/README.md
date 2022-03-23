## 4.2 循环神经网络(RNN)基础

> 此部分的内容会比较的难，希望读者可以做到“不求甚解”， 抓住核心点来进行学习。

### 4.2.1 前备知识
#### 引入

我们在[上一小节](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part2%20Deep%20Learning%20Practice/4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/4.1%20NLP%E5%9F%BA%E7%A1%80%E4%B8%8E%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)讲解过使用**潜变量**$h_t$总结过去的信息:
![image-20220206164324293](https://s2.loli.net/2022/02/06/7WIHkgB1Ty4rdjG.png)


在循环神经网络中，我们通常会更新隐藏层的状态:  $$\mathbf{h}_{t}=\phi\left(\mathbf{W}_{h h} \mathbf{h}_{t-1}+\mathbf{W}_{h x} \mathbf{x}_{t-1}+\mathbf{b}_{h}\right)$$

那么，对应输出为:  $$$\mathbf{o}_{t}=\mathbf{W}_{h_o} \mathbf{h}_{t}+\mathbf{b}_{o}$$

如下图所示：

![](https://s2.loli.net/2022/03/22/jXftGJ5naF6gpku.jpg)

#### 困惑度 (perplrxity)

*困惑度*(perplrxity)主要用途是衡量一个语言模型的好坏, 可以用平均交叉熵实现：
$$\pi=\frac{1}{n} \sum_{t=1}^{n}-\log p\left(x_{t} \mid x_{t-1}, \ldots\right)$$

其中，$p$是语言模型的预测概率，$x_t$是真实词。但由于某些历史原因NLP使用$\mathbf{exp}(\pi)$来衡量。可以看出$\mathbf{exp(\pi)} = 1$时表示完美，无穷大是最差情况。


#### 梯度裁剪
> 主要用于有效的预防**梯度爆炸**。

在梯度计算中:
* 迭代中计算这$T$个时间步上的梯度，在反向传播过程中产生长度为$O(T)$的矩阵乘法链，导致数值不稳定
* 梯度剪裁能有效预防梯度爆炸
  * 如果梯度长度超过$\theta$，那么拖影回长度$\theta$
  * 保证永远不会超过$\theta$

其公式可以表示如下:
$$\mathbf{g} \leftarrow \min \left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}$$


### RNN应用
下面这张图展现了RNN在各类NLP任务中的应用，需要注意的是，我们会人为的根据任务的不同，设定不同的输出格式:
![image-20220206232712943](https://s2.loli.net/2022/02/06/p71r6RCwI4eqaVi.png)

### 4.2.2 RNN代码实践

下面提供两种实践形式，任务都是给定前缀(prefix)一直**预测**给定文本的下一个词:
* 从零开始
* 简洁(调用pytorch API)

#### 从零开始实践

```python
%matplotlib inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35  # 每次看长为35的序列
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)  # len(vocab)为28， 词元类型默认为char
```

独热编码 one-hot

```python
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

小批量数据，其shape为(batch_size, 时间步数step), 这里就是$32 \times 35$

**注意**：由于pytorch输入的原因，我们放入`one-hot`后要转换成`(时间，批量大小，特征维度)`。 这样子访问更加$x_t$方便。

```python
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape  # 时间，批量大小，特征维度
```

**初始化模型参数**
接下来，我们初始化循环神经网络模型的模型参数：
* 隐藏单元数`num_hiddens`是一个可调的超参数。
* 当训练语言模型时，输入和输出来自相同的词表(字典)。

```python
def get_params(vocab_size, num_hiddens, device):
    """
    初始化模型参数
    :param vocab_size: 字典大小
    :param num_hiddens: 隐藏层数量
    :param device: cpu or gpu
    :return:
    """
    num_inputs = num_outputs = vocab_size # 因为输入和输出都来自于同一个字典, 因为做了一个one-hot就变成vocab_size了，输出时候是多分类问题

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01  # 最简单的初始化

    # 隐藏层的参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))  # 上一时刻的隐藏变量到下一时刻的转换
    b_h = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)
    return params
```

**RNN网络模型**
为了定义循环神经网络模型， 我们首先需要一个`init_rnn_state`函数在初始化时返回**隐藏层状态**。 这个函数的返回是一个张量，张量全用0填充， 形状为`（批量大小，隐藏单元数）`。 在后面的章节中我们将会遇到隐状态包含多个变量的情况， 而使用元组可以更容易地处理。

```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )  # 0时刻的隐藏状态，放入tuple中，由于LSTM有两个张量
```

下面的`rnn`函数定义了如何在一个时间步内计算隐藏状态和输出:

```python
def rnn(inputs, state, params):
    """
    :param inputs:(序列长度，批量大小，vocab_size)
    :param state: 初始时刻的隐藏状态
    :param params:
    :return:
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # inputs的形状：(时间步数量，批量大小，词表大小)， 证明已经转置过了
    for X in inputs:  # 按照序列遍历, X：(批量大小，词表大小) = 2 * 5
        # 当前时间步
        H = torch.tanh(torch.mm(X, W_xh)
                       + torch.mm(H, W_hh)   # 与MLP唯一不同地方
                       + b_h)
        Y = torch.mm(H, W_hq) + b_q  # 对当前时刻的预测，当前预测下一个输出是什么
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)  # ouput垂直拼接成变成一个二维的东西((批量大小 * 时间长度) * vocab_size), H为当前隐藏状态
```

定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数， 并存储从零开始实现的循环神经网络模型的参数。

```python
class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """前向传播函数"""
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

检查输出是否具有正确的形状。 例如，隐状态的维数是否保持不变。

```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

我们可以看到输出形状是`（时间步数×批量大小，词表大小）`， 而隐状态形状保持不变，即`（批量大小，隐藏单元数）`。


**预测**
首先定义预测函数来生成prefix之后的新字符， 其中的prefix是一个用户提供的包含多个字符的字符串。 在循环遍历prefix中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。 这被称为**预热**（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。

```python
def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    在prefix后面生成新字符, 预测函数
    :param prefix: 句子的给定开头
    :param num_preds: 生成多少个次
    :param net:
    :param vocab:
    :param device:
    :return:
    """
    state = net.begin_state(batch_size=1, device=device)  # 因为只对一个词做预测，所以batch_size为1
    outputs = [vocab[prefix[0]]]  # 字符串在vocab中对应的下标， 开始就是一个词元
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 放入最近那个词元
    for y in prefix[1:]:  # 预热期， 所有给定的前缀
        _, state = net(get_input(), state)  # 只做状态初始化，使用真实值
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 返回整个预测结果
```

现在我们可以测试`predict_ch8`函数。 我们将前缀指定为time traveller， 并基于这个前缀生成10个后续字符。 鉴于我们还没有训练网络，它会生成荒谬的预测结果。

```python
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

##### 梯度裁剪

```python
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

##### 训练
> 由于时间与资源关系，下面我们将省略计算的结果

```python
def train_epoch_ch8(net, train_iter, loss, updater, device,
                    use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()  # 顺序采样就detach前面就好
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```python
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

现在，我们训练循环神经网络模型。 因为我们在数据集中只使用了10000个词元， 所以模型需要更多的迭代周期来更好地收敛。


```python
# num_epochs, lr = 500, 1
# train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

#### 简洁实现

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

##### 定义模型
构造一个具有256个隐藏单元的单隐藏层的循环神经网络层`rnn_layer`

```python
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

我们使用张量来初始化隐状态，它的形状是(隐藏层数，批量大小，隐藏单元数)

```python
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。 需要强调的是，`rnn_layer`的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。

```python
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

们为一个完整的循环神经网络模型定义了一个RNNModel类。 注意，rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层。

```python
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)  # 构造自己的输出层
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

#### 训练和预测
在训练模型之前，让我们基于一个具有随机权重的模型进行预测。

```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```python
# num_epochs, lr = 500, 1
# d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

与从零开始实现相比，由于深度学习框架的高级API对代码进行了更多的优化， 该模型在较短的时间内达到了较低的困惑度。

### 总结

* 循环神经网络的输出取决于当下输入和前一时间的**隐变量**
* 应用到语言模型中时，循环神经网络根据当前词预测下一次时刻词
* 通常使用困惑度来衡量语言模型的好坏
