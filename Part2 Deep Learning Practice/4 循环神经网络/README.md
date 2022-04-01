# 4 循环神经网络(RNN)

>  从此章节开始，解决了REAME文档的**中文乱序**问题，非常感谢李沐老师做出的[贡献](https://github.com/mli/notedown)

## 4.1 NLP基础与语言模型

### 4.1.1 序列模型
很多数据都是具有时序信息的，比如说数字货币的收盘价格，如下图所示:
![image-20220316165404313](https://s2.loli.net/2022/03/16/IqeFs1KoO8pDhzm.png)

对序列数据来说：
* 数据值随时间变化而变化
* *音乐*、*语言*、*文本*、和*视频*都是连续的
* 预测明天的股价要比填补昨天遗失的股价更困难

**统计工具:**
在时间$t$观察到$x_t$， 那么得到$T$个不独立的随机变量$(x_1,...,x_T) \sim p(\mathbf{x})$

* 使用条件概率展开

$$
p(a, b)=p(a) p(b \mid a)=p(b) p(a \mid b)
$$

![image-20220204181132691](https://s2.loli.net/2022/02/04/ZuT7gXb85LvaHcN.png)

* 对条件概率建模
  * 也称为自回归模型

$$
p\left(x_{t} \mid x_{1}, \ldots x_{t-1}\right)=p\left(x_{t} \mid f\left(x_{1}, \ldots x_{t-1}\right)\right)
$$

**方案A - 马尔科夫假设**

* 假设当前当前数据只跟$\tau$个过去数据点相关

例如在过去数据上训练一个MLP
$$
p\left(x_{t} \mid x_{1}, \ldots x_{t-1}\right)=p\left(x_{t} \mid x_{t-\tau}, \ldots x_{t-1}\right)=p\left(x_{t} \mid f\left(x_{t-\tau,...,x_{t-1}}\right)\right.
$$

**方案B - 潜变量模型(latient variable)**

* 引入潜变量$h_t$来表示过去信息$h_t=f(x_1,...,x_{t-1})$
  * 这样$x_t=p(x_t|h_t)$

![image-20220204181853244](https://s2.loli.net/2022/02/04/BI7aeqKrugTLpn3.png)


下面我们来观看一些列子来直观理解使用MLP来模拟预测时间序列。

首先我们使用正弦函数和一些可加性的噪音来生成我们所需要的序列数据, 时间步长为1, 2,..., 1000。

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

将数据映射为数据对$y_t=x_t$和$\mathbf{x_t} = [x_{t-\tau}, ..., x_{t-1}]$, 这里$\tau$大小为4:

```python
tau = 4
features = torch.zeros((T - tau, tau))
features.shape
```

```python
for i in range(tau):
    features[:, i] = x[i:T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

使用一个包含两个FCN的多层感知机

```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()
```

开始训练模型

```python
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}',
              f'loss: {d2l.evaluate_loss(net, train_iter, loss)}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

预测下一个时间步：

```python
onestep_preds = net(features)
d2l.plot(
    [time, time[tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy()], 'time', 'x',
    legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3)
)
```

进行多步预测, 发现效果其实挺差的，因为每次**误差都在累计**：

```python
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

可以明显的看到，马尔科夫假设是无法预测较远的未来的！

```python
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

### 4.1.2 文本预处理与语言模型
经过上面的分析，下面我们将文本当成给一个时序序列。

#### 文本预处理
* 词元化
* 构建字典

```python
import collections
import re
from d2l import torch as d2l
```

下面我们使用`API`来下载一本书的文本，将数据集读取到由多跳文本行组成的列表当中:
> 这里我们只做暴力简单的预处理

```python
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()  # 一行一行读进来
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 标点符号和其他一些符号变成空格

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
```

##### 词元化 Tokenize
每个文本序列又拆分成一个标记列表

```python
def tokenize(lines, token='word'):
    """
    将文本拆分为单词或者字符词元
    :param lines: 文本列表
    :param token: 词元类型
    :return:
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError

tokens = tokenize(lines)
for i in range(20):  # 看前20个句子的词元
    print(tokens[i])
```

##### 词汇表 Vocabulary
构建一个字典（词汇表）：字符串类型的标记映射到从0开始的数字索引中。

用来将字符串类型的词元映射到从0开始的数字索引中。 我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计， 得到的统计结果称之为**语料**（corpus）。

然后根据每个唯一词元的出现频率，为其分配一个数字索引。 很少出现的词元通常被移除，这可以降低复杂性。

另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元`<unk>`。 我们可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（`<pad>`）； 序列开始词元（`<bos>`）； 序列结束词元（`<eos>`）。

```python
class Vocab:
    """字典"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        :param tokens: 词元
        :param min_freq: 少于多少去掉
        :param reserved_tokens: 开始和结束的token
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率从高到低排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)  # 每一个独一无二的token出现的次数
```

下面我们来构建词汇表:

```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])  # 前10个高频词
```

```python
print(vocab.idx_to_token[:10])  # 0-9索引对应的词元
```

将每一条文本行转换成一个数字索引列表：

```python
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

下面将本小节上述的所有功能都放到`load_corpus_time_machine`函数当中：
**将单词转换成转成一个数字下标的代码为**:
```python
corpus = [vocab[token] for line in tokens for token in line]
```

```python
def load_corpus_time_machine(max_tokens=-1):
    """
    返回时光机器数据集的标记索引列表和词汇表
    :param max_tokens: 最大词元数量
    :return:
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')  # 注意这里使用了字符作为词元
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]  # 将单词转换成转成一个数字下标
    if max_tokens > 0:
        corpus = corpus[:max_tokens]  # 限定最大读取词元数量
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

#### 语言模型
给定文本序列$x_1,...,x_T$，语言模型的目标是估计联合概率$p(x_1,...,x_T)$，其应用包括：
* 预处理模型(eg BERT, GPT-3)
* 生成文本，给定前面几个次，不断的使用$x_t \sim p(x_t \mid x_1,...,x_{t-1})$来生成后续文本

下面来看使用简单**计数建模**的思想：
* 假设序列长度为2，我们预测:
$$p\left(x, x^{\prime}\right)=p(x) p\left(x^{\prime} \mid x\right)=\frac{n(x)}{n} \frac{n\left(x, x^{\prime}\right)}{n(x)}$$
这里$n$是总次数，$n(x),n(x,x')$是单个单词和连续单词对的出现次数
* 我们可以拓展到长度为3的情况:
$$
                xp\left(x, x^{\prime}, x^{\prime \prime}\right)=p(x) p\left(x^{\prime} \mid x\right) p\left(x^{\prime \prime} \mid x, x^{\prime}\right)=\frac{n(x)}{n} \frac{n\left(x, x^{\prime}\right)}{n(x)} \frac{n\left(x, x^{\prime}, x^{\prime \prime}\right)}{n\left(x, x^{\prime}\right)}
$$

* 但一直将长度拓展，我们会面临一个问题: 当序列很长时间，由于文本量不够大，很可能$n(x_1,...,x_T) \leq 1$（指数级的复杂度）

使用马尔科夫假设缓解这个问题，提出**N元语法：**
一元，二元，三元语法如下所示:
$$\begin{aligned}
p\left(x_{1}, x_{2}, x_{3}, x_{4}\right) &=p\left(x_{1}\right) p\left(x_{2}\right) p\left(x_{3}\right) p\left(x_{4}\right) \\
&=\frac{n\left(x_{1}\right)}{n} \frac{n\left(x_{2}\right)}{n} \frac{n\left(x_{3}\right)}{n} \frac{n\left(x_{4}\right)}{n} \\
p\left(x_{1}, x_{2}, x_{3}, x_{4}\right) &=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{2}\right) p\left(x_{4} \mid x_{3}\right) \\
&=\frac{n\left(x_{1}\right)}{n} \frac{n\left(x_{1}, x_{2}\right)}{n\left(x_{1}\right)} \frac{n\left(x_{2}, x_{3}\right)}{n\left(x_{2}\right)} \frac{n\left(x_{3}, x_{4}\right)}{n\left(x_{3}\right)} \\
p\left(x_{1}, x_{2}, x_{3}, x_{4}\right) &=p\left(x_{1}\right) p\left(x_{2} \mid x_{1}\right) p\left(x_{3} \mid x_{1}, x_{2}\right) p\left(x_{4} \mid x_{2}, x_{3}\right)
\end{aligned}$$


下面我们引入**数据集**进行实验：

首先，进行词元化与构建字典，与之前说到过的类似

```python
import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())

corpus = [token for line in tokens for token in line]  # 这里不是索引，是单词，与上面有所不同
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]  # 前10个高频词元
```

正如我们所看到的，最高频的词看起来很无聊， 这些词通常被称为**停用词**（stop words）,我们可以画出的词频图:

```python
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

通过此图我们可以发现：词频以一种明确的方式迅速衰减。将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线。
这意味着单词的频率满足*齐普夫定律*（Zipf's law），即第$i$个最常用单词的频率$n_i$为：

$$n_i \propto \frac{1}{i^\alpha},$$

等价于
$$\log n_i = -\alpha \log i + c,$$

其中$\alpha$是刻画分布的指数，$c$是常数。这告诉我们想要通过计数统计和平滑来建模单词是不可行的，因为这样建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。那么其他的词元组合，比如二元语法、三元语法等等，又会如何呢？

我们来看看二元语法与三元语法的频率是否与一元语法的频率表现出相同的行为方式。

* 二元

```python
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

* 三元

```python
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

```python
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

可以分析出:
* 除了一元语法词，单词序列似乎也遵循齐普夫定律，尽管公式中的指数$\alpha$更小（指数的大小受序列长度的影响）。
* 词表中$n$元组的数量并没有那么大，这说明语言中存在相当多的结构，这些结构给了我们应用模型的希望。
* 很多$n$元组很少出现，这使得拉普拉斯平滑非常不适合语言建模。作为代替，我们将使用基于深度学习的模型。

##### 采样方法 mini-batch
> 这里是一个**难点**！请耐心多阅读几次代码!!!

下面展示两种生成mini_batch的方法, 都保证我们所有的数据只会**用过一次**，而之前提到过的序列模型，每个数据都会使用$\tau$次:

1. 随机生成一个小批量数据的特征和标签。在随机采样当中，每个样本都是在原始长序列上任意捕获子序列。
![](https://zh-v2.d2l.ai/_images/timemachine-5gram.svg)

```python
def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样方式生成一个小批量子序列
    :param corpus:
    :param batch_size:
    :param num_steps:每个子序列中预定义的时间步数，类似于马尔科夫假设的tao
    :return:
    """
    corpus = corpus[random.randint(0, num_steps - 1):] # 随机起始位置，减去1，因为我们需要考虑标签。前面的那几个不要了
    num_subseqs = (len(corpus) - 1) // num_steps  # 切分成num_subseqs份
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))  # 长度为num_steps的子序列的起始索引
    random.shuffle(initial_indices)  # 增强随机性, 注意与下面的第二种采样方法分开

    def data(pos):
        """返回从pos位置开始的长度为num_steps的序列"""
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

下面我们**生成一个从$0$到$34$的序列**。假设批量大小为$2$，时间步数为$5$，这意味着可以生成$\lfloor (35 - 1) / 5 \rfloor= 6$个“特征－标签”子序列对。如果设置小批量大小为$2$，我们只能得到$3$个小批量。

```python
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

2. 保证两个相邻的小批量中子序列在原始序列也是**相邻**的

在迭代过程中，除了对原始序列可以随机抽样外， 我们还可以保证两个相邻的小批量中的子序列在原始序列上也是**相邻**的。 这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序，因此称为顺序分区。

```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序分区生成一个小批量子序列
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```python
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

下面将两个采样函数包装到一个类当中:

```python
class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

最后，我们定义了一个函数`load_data_time_machine`， 它同时返回数据迭代器和词表， 因此可以与其他带有load_data前缀的函数 （如前面小节中定义的 `d2l.load_data_fashion_mnist`）类似地使用。

```python
def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

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

## 4.3 门控制单元(GRU)与长短期记忆网络(LSTM)

从本小节开始，我们将开始介绍现代循环神经网络。回顾之前小节的RNN，我们或许会发现存在一些问题：
* 我们在关注一个序列的时候，并不是每个观察值都是同等重要的，同时在RNN当中，太过久远的信息不好抽取出来。我们想只记住相关的观察，如下图所示，我们重点观察的是那只老鼠：
![](https://s2.loli.net/2022/02/07/eFQXAjbTf1PI7v8.png)
* 同时，在一些文本序列当中一些词元没有相关的观测值。 例如，在对网页内容进行情感分析时， 可能有一些辅助HTML代码与网页传达的情绪无关。 我们希望有一些机制来跳过隐状态表示中的此类词元。
* 在序列的各个部分之间存在逻辑中断的时候。 例如，书的章节之间可能会有过渡存在， 或者证券的熊市和牛市之间可能会有过渡存在。 在这种情况下，最好有一种方法来重置我们的内部状态表示。

为了解决上面所提到的一些问题，我们引入两种现代循环神经网络(`GRU`和`LSTM`)来解决。

```python
import torch
from torch import nn
from d2l import torch as d2l

# 读入数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### 4.3.1 GRU(门控制单元)
对于想只记住相关的观察，我们可以:
* 能关注的机制（更新门）: update gate
    * 允许我们控制新状态中有多少个是旧状态的副本。
* 能遗忘的机制（重置门）: reset gate
    * 允许我们控制“可能还想记住”的过去状态的数量。

#### 门
更新门和重置门的公式与图示如下:

$$
\begin{aligned}
&\boldsymbol{R}_{t}=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x r}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h r}+\boldsymbol{b}_{r}\right) \\
&\boldsymbol{Z}_{t}=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x z}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h z}+\boldsymbol{b}_{z}\right)
\end{aligned}
$$

![](https://zh-v2.d2l.ai/_images/gru-1.svg)

#### 候选隐藏状态 (Candidate Hidden State)
用来生成真正的隐藏状态，其公式与图示如下:
$$
\tilde{\boldsymbol{H}}_{t}=\tanh \left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}+\left(\boldsymbol{R}_{t} \odot \boldsymbol{H}_{t-1}\right) \boldsymbol{W}_{h h}+\boldsymbol{b}_{h}\right)
$$

![](https://zh-v2.d2l.ai/_images/gru-2.svg)


#### 隐状态 (Hidden State)
$$
\boldsymbol{H}_{t}=\boldsymbol{Z}_{t} \odot \boldsymbol{H}_{t-1}+\left(1-Z_{t}\right) \odot \tilde{\boldsymbol{H}}_{t}
$$

![](https://zh-v2.d2l.ai/_images/gru-3.svg)



#### GRU总结
其计算过程与模型架构如下所示:

$$
\begin{aligned}
&\boldsymbol{R}_{t}=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x r}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h r}+b_{r}\right) \\
&Z_{t}=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x z}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h z}+b_{z}\right) \\
&\tilde{\boldsymbol{H}}_{t}=\tanh \left(\boldsymbol{X}_{t} \boldsymbol{W}_{x h}+\left(\boldsymbol{R}_{t} \odot \boldsymbol{H}_{t-1}\right) \boldsymbol{W}_{h h}+\boldsymbol{b}_{h}\right) \\
&\boldsymbol{H}_{t}=\boldsymbol{Z}_{t} \odot \boldsymbol{H}_{t-1}+\left(1-\boldsymbol{Z}_{t}\right) \odot \tilde{\boldsymbol{H}}_{t}
\end{aligned}
$$

![](https://zh-v2.d2l.ai/_images/gru-3.svg)

#### 代码实践
下面我们通过两种方式进行实践:
* 从零开始实现
* 调用`Pytorch` API。

*注意*:此部分与[4.2](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part2%20Deep%20Learning%20Practice/4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/4.2%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%A5%E9%97%A8)小节内容较为相似，如果代码中遇到不懂的可以回去看[4.2](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part2%20Deep%20Learning%20Practice/4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/4.2%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%85%A5%E9%97%A8)小节


##### ① 从零开始实现

* 初始化模型参数

从标准差为0.01的高斯分布中提取权重， 并将偏置项设为0，超参数num_hiddens定义隐藏单元的数量， 并且实例化与更新门、重置门、候选隐状态和输出层相关的所有权重和偏置。

```python
def get_params(vocab_size, num_hiddens, device):
    """
    初始化模型参数
    :param vocab_size:
    :param num_hiddens:
    :param device:
    :return:
    """
    num_inputs = num_outputs = vocab_size  # 因为是one-hot的

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        """实例化门与候选隐状态参数"""
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

* 定义模型

定义隐状态的初始化函数:

```python
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)
```

定义门控循环单元模型， 模型的架构与基本的循环神经网络单元是相同的， 只是权重更新公式更为复杂。

```python
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

* 训练与预测

分别打印输出训练集的困惑度， 以及前缀“time traveler”和“traveler”的预测序列上的困惑度。

```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

##### ② Pytorch API实现

高级API包含了前文介绍的所有配置细节， 所以我们可以直接实例化门控循环单元模型。

同时下面这段代码的运行速度要快得多， 因为它使用的是编译好的运算符而不是`Python`来处理之前阐述的许多细节, 同时优化了很多大矩阵乘法的细节。

```python
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

### 4.3.2 LSTM(长短期记忆网络)
虽然LSTM放在了GRU的后面来进行介绍，但LSTM的发明远远早于GRU。它是上世纪90年代发明的。

其设计的核心如下:
* 忘记门: 将值朝0减少
* 输入门: 决定不是忽略掉输入数据
* 输出门: 决定是不是使用隐状态

”三门“可用如下公式进行计算:
$$
\begin{aligned}
\boldsymbol{I}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x i}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h i}+\boldsymbol{b}_{i}\right) \\
\boldsymbol{F}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x f}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h f}+\boldsymbol{b}_{f}\right) \\
\boldsymbol{O}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x o}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h o}+\boldsymbol{b}_{o}\right)
\end{aligned}
$$

![](https://zh-v2.d2l.ai/_images/lstm-0.svg)

#### 候选记忆单元
$$
\tilde{\boldsymbol{C}}_{t}=\tanh \left(\boldsymbol{X}_{t} \boldsymbol{W}_{x c}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h c}+\boldsymbol{b}_{c}\right)
$$

![](https://zh-v2.d2l.ai/_images/lstm-2.svg)

#### 隐状态

使其在[-1, 1]之间：

$$
\boldsymbol{H}_{t}=\boldsymbol{O}_{t} \odot \tanh \left(\boldsymbol{C}_{t}\right)
$$

![](https://zh-v2.d2l.ai/_images/lstm-3.svg)

#### 汇总公式
$$
\begin{aligned}
\boldsymbol{I}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x i}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h i}+\boldsymbol{b}_{i}\right) \\
\boldsymbol{F}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x f}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h f}+\boldsymbol{b}_{f}\right) \\
\boldsymbol{O}_{t} &=\sigma\left(\boldsymbol{X}_{t} \boldsymbol{W}_{x o}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h o}+\boldsymbol{b}_{o}\right) \\
\tilde{\boldsymbol{C}}_{t} &=\tanh \left(\boldsymbol{X}_{t} \boldsymbol{W}_{x c}+\boldsymbol{H}_{t-1} \boldsymbol{W}_{h c}+\boldsymbol{b}_{c}\right) \\
\boldsymbol{C}_{t} &=\boldsymbol{F}_{t} \odot \boldsymbol{C}_{t-1}+\boldsymbol{I}_{t} \odot \tilde{\boldsymbol{C}}_{t} \\
\boldsymbol{H}_{t} &=\boldsymbol{O}_{t} \odot \tanh \left(\boldsymbol{C}_{t}\right)
\end{aligned}
$$

#### 代码实践
* 从零开始
* 简洁实现

##### ①从零开始实现

* 初始化参数

```python
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

* 模型定义

```python
def init_lstm_state(batch_size, num_hiddens, device):
    """H和C都要初始化，但形状是一样的"""
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

* 训练与预测

```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

##### ②简洁实现(Pytorch API)

```python
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

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

## 4.5 编码器与解码器架构

语言模型是自然语言处理的关键， 而**机器翻译**是语言模型最成功的基准测试。 因为机器翻译正是将输入序列转换成输出序列的序列转换模型（sequence transduction）的核心问题。 下面我们引入新的机器翻译(英语翻法语)数据集来讲解编码器与解码器架构

```python
import os
import torch
from d2l import torch as d2l
from torch import nn
```

### 4.5.1 机器翻译与数据集

首先我们下载(英语 --> 法语)数据集:

```python
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```


* 文本预处理

原始文本数据需要经过几个简单**预处理**步骤:
1. 标点符号前面空格。
2. 我们用空格代替不间断空格（non-breaking space）。
3. 使用小写字母替换大写字母，并在单词和标点符号之间插入空格。

```python
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' ' # 标点前面空格

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()   # 半角全角空格全部变成单个空格
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

* 词元化

机器翻译中，我们更喜欢**单词级词元化**: （最先进的模型可能使用更高级的词元化技术）。

1. `tokenize_nmt`函数对前`num_examples`个文本序列对进行词元， 其中每个词元要么是一个词，要么是一个标点符号。

2. 此函数返回两个词元列表：`source`和`target`： `source[i]`是源语言（这里是英语）第i个文本序列的词元列表， `target[i]`是目标语言（这里是法语）第i个文本序列的词元列表。

```python
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

* 可视化

绘制每个文本序列所包含的词元数量的直方图。 在这个简单的“英－法”数据集中，大多数文本序列的词元数量少于20个。

```python
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target);
```

* **构建词汇表**

由于机器翻译数据集由语言对组成， 因此我们可以分别为源语言和目标语言构建两个词表。


使用单词级词元化时，词表大小将明显*大于*使用字符级词元化时的词表大小。 为了缓解这一问题，这里我们将:

* 出现次数少于2次的低频率词元 视为相同的未知（`<unk>`）词元。
* 小批量时用于将序列填充到相同长度的填充词元（`<pad>`）。
* 序列的开始词元（`<bos>`）。
* 结束词元（`<eos>`）。

上述这些特殊词元在NLP的众多任务中都非常常见！

```python
src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

* 截断文本序列

下面的`truncate_pad`函数将根据序列文本的固定长度截断或填充文本序列。

当然目前有分片的技术可以避免文本截断，感兴趣的读者可以自行了解

```python
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

* 小批量数据集训练

```python
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]  # 每个句子结束为eos
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)  # 实际上有多长，不算入padding
    return array, valid_len
```

* 载入数据集

```python
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

下面我们读出“英语－法语”数据集中的第一个小批量数据：

```python
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
```

### 4.5.2 Encoder-Decoder 架构

首先在讲解编码器解码器之前，我们对`CNN`进行重新的观察, 我们把`CNN`也理解为一个`encode`和`decode`的过程，其示意图如下所示:

![image-20220209162714848](https://s2.loli.net/2022/02/09/TQ7A2mYlc1e9hjo.png)


可以观察到：
* 编码器（encoder）:将输入编程成中间表达形式(特征)
  * 将文本表示成向量
* 解码器（decoder）:将中间表示解码成输出
  * 向量表示成输出

![image-20220209162907917](https://s2.loli.net/2022/02/09/7rPTW2kvnxFiBSR.png)


而在这里，我们要介绍的编码器-解码器架构如下（一个模型分成**两块**）:
* 编码器处理输入
* 解码器处理输出

![../_images/encoder-decoder.svg](https://zh-v2.d2l.ai/_images/encoder-decoder.svg)

下面进行代码实践。

* 编码器

```python
class Encoder(nn.Module):
    """编码器-解码器结构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

* 解码器

有一个初始状态，编码器的东西要传给它

```python
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

* 合并架构

总而言之，“编码器-解码器”架构包含了一个编码器和一个解码器， 并且还拥有可选的额外的参数。 在前向传播中，编码器的输出用于生成编码状态， 这个状态又被解码器作为其**输入**的一部分。

```python
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

可以看到我们这里只是简单写出了`encoder-decoder`的基类，在后面小节我们将再深入说说他是如何使用的。

## 4.6 序列到序列(Seq2Seq)学习模型
 机器翻译中的输入序列和输出序列都是长度可变的。 为了解决这类问题，我们在[4.5](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part2%20Deep%20Learning%20Practice/4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/4.5%20%E7%BC%96%E7%A0%81%E5%99%A8%E4%B8%8E%E8%A7%A3%E7%A0%81%E5%99%A8%E6%9E%B6%E6%9E%84)小节中 设计了一个通用的”编码器－解码器“架构。 本小节，我们将使用两个循环神经网络的编码器和解码器， 并将其应用于序列到序列（Seq2Seq）类的学习任务。

动机：
* 给定一个源语言的句子，自动翻译成目标语言
* 这两个句子可以有不同的长度

### 4.6.1 模型架构
> Encoder-decoder架构

* 编码器是一个RNN，读取输入句子
  * 可以是双向（一般双向可以用作encoder，看到整个句子）
* 解码器使用另外一个RNN来输出

其模型架构图如下图所示：
![image-20220210101638258](https://s2.loli.net/2022/02/10/d1SRiOhQYo7U3xr.png)

![image-20220210102839313](https://s2.loli.net/2022/02/10/S2l9AthZPLweMRp.png)

从上图可以看到:
* 编码器是没有输出的RNN
* 编码器最后时间步的隐状态用作解码器的初始隐状态

因此在**训练**的时候:
* 机器翻译的时候
    * 训练时解码器使用目标句子作为输出
* 推理的时候
    * 只能给出上一时刻的输出

### 4.6.2 衡量生成序列的好坏的BLEU

* $p_n$是预测中所有n-gram的精度
  * 标签序列ABCDEF和预测序列ABBCD
    * $p_1=4/5,p_2=3/4,p_3=1/3,p_4=0$
* BLEU定义
  * 越大越好！

$$
\exp \left(\min \left(0,1-\frac{1 \mathrm{en}_{\text {label }}}{\operatorname{len}_{\text {pred }}}\right) \prod_{n=1}^{k} p_{n}^{1 / 2^{n}}\right.
$$

$\mathbf{exp(...)}$为惩罚过短的预测,$\Pi...$表示长匹配有效权重

### 4.6.3 代码实践

使用[4.5](https://github.com/Gary-code/Machine-Learning-Park/tree/main/Part2%20Deep%20Learning%20Practice/4%20%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/4.5%20%E7%BC%96%E7%A0%81%E5%99%A8%E4%B8%8E%E8%A7%A3%E7%A0%81%E5%99%A8%E6%9E%B6%E6%9E%84)小节讲到的机器翻译数据集（英语 -> 法语）进行训练和预测。

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
```

* 编码器实现

```python
class Seq2SeqEncoder(d2l.Encoder):
    """Seq2Seq的编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
        # 编码器不需要输出层

    def forward(self, X, *args):
        """
        :param self:
        :param X: [batch_size, num_steps, vocab_size]
        :param args:
        :return:
        output[num_steps, batch_size, num_hiddens]
        state:[num_layers, batch_size, num_hiddens]
        """
        X = self.embedding(X)  # [:, :, vocab_size] -> [:, :, embed_size]
        X = X.permute(1, 0, 2)  # [num_steps, batch_size, embed_size]
        output, state = self.rnn(X)  # 如果没有提及状态，默认为0
        return output, state
```

测试编码器

```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape, state.shape
```

* 解码器的实现:

```python
class Seq2SeqDecoder(d2l.Decoder):
    """Seq2Seq解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 输出层

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]  # [output, state] 1就代表state

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)  # 变为[num_steps, batch_size, embed_size]
        # 广播context，使其具有与X相同的num_steps
        print(f'X',X.shape)
        print(f'state',state.shape)
        print(f'state1',state[-1].repeat(X.shape[0], 1, 1).shape)
        context = state[-1].repeat(X.shape[0], 1, 1)  # 取出最近的一个隐藏状态
        X_and_context = torch.cat((X, context), 2)
        print(f'X_and_Context', X_and_context.shape)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state
```

测试解码器

```python
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

* 想通过**0值**屏蔽不相关的项

```python
def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
    :param X:
    :param valid_len:
    :param value:
    :return:
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    print(f'mask', mask)
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

我们还可以使用此函数屏蔽最后几个轴上的所有项。如果愿意，也可以使用指定的非零值来替换这些项。

```python
X = torch.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

通过扩展softmax交叉熵损失函数来遮蔽不相关的预测。注意：填充不参与计算。

最初，所有预测词元的掩码都设置为1。 一旦给定了有效长度，填充词元对应的掩码将被设置为0。 最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。

**注意:** `unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)` pytorch要求预测的维度需要放在中间

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        """
        :param pred:  (batch_size,num_steps,vocab_size)
        :param label:  (batch_size,num_steps)
        :param valid_len: (batch_size,)
        :return:
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)  # 有效的1无效的0
        self.reduction='none'  # 不要求mean之类的
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)  # pytorch要求预测的维度需要放在中间
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # 对每个句子取一个平均
        return weighted_loss  # 对每一个样本（句子）返回一个loss
```

我们可以创建三个相同的序列来进行代码健全性检查， 然后分别指定这些序列的有效长度为4、2和0。 结果就是，第一个序列的损失应为第二个序列的两倍，而第三个序列的损失应为零。

```python
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

* 训练

```python
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

由于资源优先这里不跑训练的代码了，读者可自行将`train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)`的注释去掉运行查看训练效果。

```python
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
# train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

## 4.7 束搜索(Beam Search)

> 引入

在`seq2seq`中我们使用了贪心搜索来预测序列，是将当前时刻预测概率最大的词输出，但贪心很可能不是最优的，因为对于很多序列任务，当前选择对未来是有影响的。

### 4.7.1 贪心搜索
一旦输出序列包含了`<eos>`或者达到其最大长度$T^{\prime}$，则输出完成。 在每个时间步，贪心搜索选择具有最高条件概率的词元, 如下图所示:
![](https://zh-v2.d2l.ai/_images/s2s-prob1.svg)


上图中, 预测输出序列“A”、“B”、“C”和“<eos>”。 这个输出序列的条件概率是:
$$
0.5 \times 0.4 \times 0.4 \times 0.6=0.048
$$

现实中，最优序列（optimal sequence）应该是最大化$\prod_{t^{\prime}=1}^{T^{\prime}} P\left(y_{t^{\prime}} \mid y_{1}, \ldots, y_{t^{\prime}-1}, \mathbf{c}\right)$值的输出序列，这是基于输入序列生成输出序列的条件概率。 然而，贪心搜索无法保证得到最优序列。

### 4.7.2 穷举搜索
> 复杂度非常非常高！！！不可行！！!

通过下面公式的计算，可以看出来复杂度非常高。
$$
若n=10000, \quad T=10 则\quad n^{T}=10^{40}
$$

### 4.7.3 束搜索 (Beam Search)

* 保存最好的$k$个候选
* 在每个时刻，对每个候选新加一项($n$种可能)，在$kn$个选项中选出最好的$k$个

![](https://zh-v2.d2l.ai/_images/beam-search.svg)

* 时间复杂度$O(knT)$
  * $k=5,n = 10000,T=10,knT= 5 \times 10^5$
* 每一个候选的最终分数为:

$$
\frac{1}{L^{\alpha}} \log p\left(y_{1}, \ldots, y_{L}\right)=\frac{1}{L^{\alpha}} \sum_{t^{\prime}=1}^{L} \log p\left(y_{t^{\prime}} \mid y_{1}, \ldots, y_{t^{'}}\right)
$$

* 通常，$\alpha=0.75$

### 4.7.4 总结

束搜索在每次搜索时保存$k$个最好的候选，特殊的:

* $k=1$时是贪心搜索
* $k=n$时是穷举搜索