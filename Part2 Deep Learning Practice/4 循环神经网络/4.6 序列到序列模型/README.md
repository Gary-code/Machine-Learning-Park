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
