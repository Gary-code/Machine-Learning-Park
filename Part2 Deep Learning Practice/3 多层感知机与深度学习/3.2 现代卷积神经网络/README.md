## 3.2 现代卷积神经网络
现代卷积神经网络的研究中下面每一个模型都曾一度占据主导地位，其中许多模型都是ImageNet竞赛的优胜者。ImageNet竞赛自2010年以来，一直是计算机视觉中**监督学习**进展的指向标。其包括:
* LeNet: 最”远古“的神经网络。
* AlexNet。它是第一个在大规模视觉竞赛中击败传统计算机视觉模型的大型神经网络。
*
使用重复块的网络（VGG）。它利用许多重复的神经网络块。
* 网络中的网络（NiN）。它重复使用由卷积层和$1 \times
1$卷积层（用来代替全连接层）来构建深层网络。
*
含并行连结的网络（GoogLeNet）。它使用并行连结的网络，通过不同窗口大小的卷积层和最大池化层来并行抽取信息。
*
残差网络（ResNet）。它通过残差块构建跨层的数据通道，是计算机视觉中最流行的体系架构。
*
稠密连接网络（DenseNet）。它的计算成本很高，但给我们带来了更好的效果。

*温馨提示: 本小节内容较多，读者可根据自己需要来进行阅读。*

### 3.2.1 LeNet
这个模型是由AT&T贝尔实验室的研究员Yann LeCun在1989年提出的（并以其命名），目的是识别图像中的手写数字。
当时，Yann LeCun发表了第一篇通过反向传播成功训练卷积神经网络的研究，这项工作代表了十多年来神经网络研究开发的成果。
当时，LeNet取得了与支持向量机（support vector machines）性能相媲美的成果，成为监督学习的主流方法。
LeNet被广泛用于自动取款机（ATM）机中，帮助识别处理支票的数字。 时至今日，一些自动取款机仍在运行Yann LeCun和他的同事Leon
Bottou在上世纪90年代写的代码。

总体来看，LeNet（LeNet-5）由两个部分组成：
* 卷积编码器：由**两个卷积层**组成
*
全连接层密集块：由**三个全连接层**组成
每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和平均池化层。请注意，虽然ReLU和最大池化层更有效，但它们在20世纪90年代还没有出现。每个卷积层使用$5
\times
5$卷积核和一个sigmoid激活函数。这些层将输入映射到多个二维特征输出，通常同时增加通道的数量。第一卷积层有6个输出通道，而第二个卷积层有16个输出通道。每个$2
\times 2$池操作（步骤2）通过空间下采样将维数减少4倍。卷积的输出形状由批量大小、通道数、高度、宽度决定。如下图所示:
![](https://zh-v2.d2l.ai/_images/lenet.svg)
为了将卷积块的输出传递给稠密块，我们必须在小批量中展平每个样本。换言之，我们将这个四维输入转换成全连接层所期望的二维输入。这里的二维表示的第一个维度索引小批量中的样本，第二个维度给出每个样本的平面向量表示。LeNet的稠密块有三个全连接层，分别有120、84和10个输出。因为我们在执行分类任务，所以输出层的10维对应于最后输出结果的数量(识别0-9的数字)。
下面我们通过Pytorch代码，使用一个Sequential块即可把上面所描述的层全部连接起来。

对原始模型做了一点小改动，去掉了最后一层的高斯激活。

```{.python .input  n=4}
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

下面，我们将一个大小为$28 \times 28$的单通道（黑白）图像通过LeNet。通过在每一层打印输出的形状，我们可以检查模型是否与下图保持一致:
![](https://zh-v2.d2l.ai/_images/lenet-vert.svg)

```{.python .input  n=5}
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Conv2d output shape: \t torch.Size([1, 6, 28, 28])\nSigmoid output shape: \t torch.Size([1, 6, 28, 28])\nAvgPool2d output shape: \t torch.Size([1, 6, 14, 14])\nConv2d output shape: \t torch.Size([1, 16, 10, 10])\nSigmoid output shape: \t torch.Size([1, 16, 10, 10])\nAvgPool2d output shape: \t torch.Size([1, 16, 5, 5])\nFlatten output shape: \t torch.Size([1, 400])\nLinear output shape: \t torch.Size([1, 120])\nSigmoid output shape: \t torch.Size([1, 120])\nLinear output shape: \t torch.Size([1, 84])\nSigmoid output shape: \t torch.Size([1, 84])\nLinear output shape: \t torch.Size([1, 10])\n"
 }
]
```

#### 训练模型
下面我们通过训练模型进行实战，我们选取Fashion-MNIST数据集
* Fashion-MNIST: 含有非数字的图片
*
原始MNIST: 只含有数字图片

为了演示方便，同样使用d2l库，并且我的本机环境采用CPU进行训练，但是如果有条件，可以尝试GPU来加快训练。
下面代码在GPU与CPU上均可跑通:

* 加载数据集

```{.python .input  n=6}
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data\\FashionMNIST\\raw\\train-images-idx3-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "83eba22b114a43fc8006ecb1f0720ef8",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/26421880 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data\\FashionMNIST\\raw\\train-images-idx3-ubyte.gz to ../data\\FashionMNIST\\raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data\\FashionMNIST\\raw\\train-labels-idx1-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "c6ec91757df34351b7f79dfc60277948",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/29515 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data\\FashionMNIST\\raw\\train-labels-idx1-ubyte.gz to ../data\\FashionMNIST\\raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data\\FashionMNIST\\raw\\t10k-images-idx3-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "1be475dd311345868912c22af40cac9b",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/4422102 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data\\FashionMNIST\\raw\\t10k-images-idx3-ubyte.gz to ../data\\FashionMNIST\\raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data\\FashionMNIST\\raw\\t10k-labels-idx1-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "876f7b580a1f4bc789d5ec974aff3288",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/5148 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data\\FashionMNIST\\raw\\t10k-labels-idx1-ubyte.gz to ../data\\FashionMNIST\\raw\n\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "D:\\tensorflow1\\lib\\site-packages\\torchvision\\datasets\\mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\\torch\\csrc\\utils\\tensor_numpy.cpp:180.)\n  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)\n"
 }
]
```



由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中。

* 计算模型精度

```{.python .input  n=7}
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """ 使用GPU计算模型所在数据集的精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总浏览数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

下面我们开始训练:
我们使用Xavier随机初始化模型参数。 与全连接层一样，我们使用交叉熵损失函数和小批量随机梯度下降。

```{.python .input  n=8}
def train(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input  n=9}
lr, num_epochs = 0.9, 10
train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.465, train acc 0.827, test acc 0.828\n5740.3 examples/sec on cpu\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (http://matplotlib.org/) -->\r\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 252.64375 184.15625\" width=\"252.64375pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <defs>\r\n  <style type=\"text/css\">\r\n*{stroke-linecap:butt;stroke-linejoin:round;}\r\n  </style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 184.15625 \r\nL 252.64375 184.15625 \r\nL 252.64375 -0 \r\nL 0 -0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 43.78125 146.6 \r\nL 239.08125 146.6 \r\nL 239.08125 10.7 \r\nL 43.78125 10.7 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 65.48125 146.6 \r\nL 65.48125 10.7 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_2\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"mc5caa29d08\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"65.48125\" xlink:href=\"#mc5caa29d08\" y=\"146.6\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 2 -->\r\n      <defs>\r\n       <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-32\"/>\r\n      </defs>\r\n      <g transform=\"translate(62.3 161.198437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-32\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_3\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 108.88125 146.6 \r\nL 108.88125 10.7 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"108.88125\" xlink:href=\"#mc5caa29d08\" y=\"146.6\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 4 -->\r\n      <defs>\r\n       <path d=\"M 37.796875 64.3125 \r\nL 12.890625 25.390625 \r\nL 37.796875 25.390625 \r\nz\r\nM 35.203125 72.90625 \r\nL 47.609375 72.90625 \r\nL 47.609375 25.390625 \r\nL 58.015625 25.390625 \r\nL 58.015625 17.1875 \r\nL 47.609375 17.1875 \r\nL 47.609375 0 \r\nL 37.796875 0 \r\nL 37.796875 17.1875 \r\nL 4.890625 17.1875 \r\nL 4.890625 26.703125 \r\nz\r\n\" id=\"DejaVuSans-34\"/>\r\n      </defs>\r\n      <g transform=\"translate(105.7 161.198437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-34\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_5\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 152.28125 146.6 \r\nL 152.28125 10.7 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"152.28125\" xlink:href=\"#mc5caa29d08\" y=\"146.6\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 6 -->\r\n      <defs>\r\n       <path d=\"M 33.015625 40.375 \r\nQ 26.375 40.375 22.484375 35.828125 \r\nQ 18.609375 31.296875 18.609375 23.390625 \r\nQ 18.609375 15.53125 22.484375 10.953125 \r\nQ 26.375 6.390625 33.015625 6.390625 \r\nQ 39.65625 6.390625 43.53125 10.953125 \r\nQ 47.40625 15.53125 47.40625 23.390625 \r\nQ 47.40625 31.296875 43.53125 35.828125 \r\nQ 39.65625 40.375 33.015625 40.375 \r\nz\r\nM 52.59375 71.296875 \r\nL 52.59375 62.3125 \r\nQ 48.875 64.0625 45.09375 64.984375 \r\nQ 41.3125 65.921875 37.59375 65.921875 \r\nQ 27.828125 65.921875 22.671875 59.328125 \r\nQ 17.53125 52.734375 16.796875 39.40625 \r\nQ 19.671875 43.65625 24.015625 45.921875 \r\nQ 28.375 48.1875 33.59375 48.1875 \r\nQ 44.578125 48.1875 50.953125 41.515625 \r\nQ 57.328125 34.859375 57.328125 23.390625 \r\nQ 57.328125 12.15625 50.6875 5.359375 \r\nQ 44.046875 -1.421875 33.015625 -1.421875 \r\nQ 20.359375 -1.421875 13.671875 8.265625 \r\nQ 6.984375 17.96875 6.984375 36.375 \r\nQ 6.984375 53.65625 15.1875 63.9375 \r\nQ 23.390625 74.21875 37.203125 74.21875 \r\nQ 40.921875 74.21875 44.703125 73.484375 \r\nQ 48.484375 72.75 52.59375 71.296875 \r\nz\r\n\" id=\"DejaVuSans-36\"/>\r\n      </defs>\r\n      <g transform=\"translate(149.1 161.198437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-36\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_7\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 195.68125 146.6 \r\nL 195.68125 10.7 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"195.68125\" xlink:href=\"#mc5caa29d08\" y=\"146.6\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 8 -->\r\n      <defs>\r\n       <path d=\"M 31.78125 34.625 \r\nQ 24.75 34.625 20.71875 30.859375 \r\nQ 16.703125 27.09375 16.703125 20.515625 \r\nQ 16.703125 13.921875 20.71875 10.15625 \r\nQ 24.75 6.390625 31.78125 6.390625 \r\nQ 38.8125 6.390625 42.859375 10.171875 \r\nQ 46.921875 13.96875 46.921875 20.515625 \r\nQ 46.921875 27.09375 42.890625 30.859375 \r\nQ 38.875 34.625 31.78125 34.625 \r\nz\r\nM 21.921875 38.8125 \r\nQ 15.578125 40.375 12.03125 44.71875 \r\nQ 8.5 49.078125 8.5 55.328125 \r\nQ 8.5 64.0625 14.71875 69.140625 \r\nQ 20.953125 74.21875 31.78125 74.21875 \r\nQ 42.671875 74.21875 48.875 69.140625 \r\nQ 55.078125 64.0625 55.078125 55.328125 \r\nQ 55.078125 49.078125 51.53125 44.71875 \r\nQ 48 40.375 41.703125 38.8125 \r\nQ 48.828125 37.15625 52.796875 32.3125 \r\nQ 56.78125 27.484375 56.78125 20.515625 \r\nQ 56.78125 9.90625 50.3125 4.234375 \r\nQ 43.84375 -1.421875 31.78125 -1.421875 \r\nQ 19.734375 -1.421875 13.25 4.234375 \r\nQ 6.78125 9.90625 6.78125 20.515625 \r\nQ 6.78125 27.484375 10.78125 32.3125 \r\nQ 14.796875 37.15625 21.921875 38.8125 \r\nz\r\nM 18.3125 54.390625 \r\nQ 18.3125 48.734375 21.84375 45.5625 \r\nQ 25.390625 42.390625 31.78125 42.390625 \r\nQ 38.140625 42.390625 41.71875 45.5625 \r\nQ 45.3125 48.734375 45.3125 54.390625 \r\nQ 45.3125 60.0625 41.71875 63.234375 \r\nQ 38.140625 66.40625 31.78125 66.40625 \r\nQ 25.390625 66.40625 21.84375 63.234375 \r\nQ 18.3125 60.0625 18.3125 54.390625 \r\nz\r\n\" id=\"DejaVuSans-38\"/>\r\n      </defs>\r\n      <g transform=\"translate(192.5 161.198437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-38\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_9\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 239.08125 146.6 \r\nL 239.08125 10.7 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_10\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"239.08125\" xlink:href=\"#mc5caa29d08\" y=\"146.6\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 10 -->\r\n      <defs>\r\n       <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-31\"/>\r\n       <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-30\"/>\r\n      </defs>\r\n      <g transform=\"translate(232.71875 161.198437)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-31\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_6\">\r\n     <!-- epoch -->\r\n     <defs>\r\n      <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-65\"/>\r\n      <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-70\"/>\r\n      <path d=\"M 30.609375 48.390625 \r\nQ 23.390625 48.390625 19.1875 42.75 \r\nQ 14.984375 37.109375 14.984375 27.296875 \r\nQ 14.984375 17.484375 19.15625 11.84375 \r\nQ 23.34375 6.203125 30.609375 6.203125 \r\nQ 37.796875 6.203125 41.984375 11.859375 \r\nQ 46.1875 17.53125 46.1875 27.296875 \r\nQ 46.1875 37.015625 41.984375 42.703125 \r\nQ 37.796875 48.390625 30.609375 48.390625 \r\nz\r\nM 30.609375 56 \r\nQ 42.328125 56 49.015625 48.375 \r\nQ 55.71875 40.765625 55.71875 27.296875 \r\nQ 55.71875 13.875 49.015625 6.21875 \r\nQ 42.328125 -1.421875 30.609375 -1.421875 \r\nQ 18.84375 -1.421875 12.171875 6.21875 \r\nQ 5.515625 13.875 5.515625 27.296875 \r\nQ 5.515625 40.765625 12.171875 48.375 \r\nQ 18.84375 56 30.609375 56 \r\nz\r\n\" id=\"DejaVuSans-6f\"/>\r\n      <path d=\"M 48.78125 52.59375 \r\nL 48.78125 44.1875 \r\nQ 44.96875 46.296875 41.140625 47.34375 \r\nQ 37.3125 48.390625 33.40625 48.390625 \r\nQ 24.65625 48.390625 19.8125 42.84375 \r\nQ 14.984375 37.3125 14.984375 27.296875 \r\nQ 14.984375 17.28125 19.8125 11.734375 \r\nQ 24.65625 6.203125 33.40625 6.203125 \r\nQ 37.3125 6.203125 41.140625 7.25 \r\nQ 44.96875 8.296875 48.78125 10.40625 \r\nL 48.78125 2.09375 \r\nQ 45.015625 0.34375 40.984375 -0.53125 \r\nQ 36.96875 -1.421875 32.421875 -1.421875 \r\nQ 20.0625 -1.421875 12.78125 6.34375 \r\nQ 5.515625 14.109375 5.515625 27.296875 \r\nQ 5.515625 40.671875 12.859375 48.328125 \r\nQ 20.21875 56 33.015625 56 \r\nQ 37.15625 56 41.109375 55.140625 \r\nQ 45.0625 54.296875 48.78125 52.59375 \r\nz\r\n\" id=\"DejaVuSans-63\"/>\r\n      <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 75.984375 \r\nL 18.109375 75.984375 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-68\"/>\r\n     </defs>\r\n     <g transform=\"translate(126.203125 174.876562)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-65\"/>\r\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-70\"/>\r\n      <use x=\"125\" xlink:href=\"#DejaVuSans-6f\"/>\r\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-63\"/>\r\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-68\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_11\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 145.753852 \r\nL 239.08125 145.753852 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_12\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"mc3c83866e9\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#mc3c83866e9\" y=\"145.753852\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 0.0 -->\r\n      <defs>\r\n       <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-2e\"/>\r\n      </defs>\r\n      <g transform=\"translate(20.878125 149.553071)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-30\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_13\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 118.446872 \r\nL 239.08125 118.446872 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#mc3c83866e9\" y=\"118.446872\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_8\">\r\n      <!-- 0.5 -->\r\n      <defs>\r\n       <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-35\"/>\r\n      </defs>\r\n      <g transform=\"translate(20.878125 122.246091)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-30\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-35\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_15\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 91.139892 \r\nL 239.08125 91.139892 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_16\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#mc3c83866e9\" y=\"91.139892\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 1.0 -->\r\n      <g transform=\"translate(20.878125 94.939111)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-31\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_17\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 63.832912 \r\nL 239.08125 63.832912 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_18\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#mc3c83866e9\" y=\"63.832912\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_10\">\r\n      <!-- 1.5 -->\r\n      <g transform=\"translate(20.878125 67.63213)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-31\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-35\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_19\">\r\n      <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 36.525931 \r\nL 239.08125 36.525931 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_20\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#mc3c83866e9\" y=\"36.525931\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 2.0 -->\r\n      <g transform=\"translate(20.878125 40.32515)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-32\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_12\">\r\n     <!-- None -->\r\n     <defs>\r\n      <path d=\"M 9.8125 72.90625 \r\nL 23.09375 72.90625 \r\nL 55.421875 11.921875 \r\nL 55.421875 72.90625 \r\nL 64.984375 72.90625 \r\nL 64.984375 0 \r\nL 51.703125 0 \r\nL 19.390625 60.984375 \r\nL 19.390625 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-4e\"/>\r\n      <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-6e\"/>\r\n     </defs>\r\n     <g transform=\"translate(14.798437 91.695312)rotate(-90)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-4e\"/>\r\n      <use x=\"74.804688\" xlink:href=\"#DejaVuSans-6f\"/>\r\n      <use x=\"135.986328\" xlink:href=\"#DejaVuSans-6e\"/>\r\n      <use x=\"199.365234\" xlink:href=\"#DejaVuSans-65\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_21\">\r\n    <path clip-path=\"url(#pc51e788df5)\" d=\"M 26.42125 16.877273 \r\nL 30.76125 18.042339 \r\nL 35.10125 18.551131 \r\nL 39.44125 18.860483 \r\nL 43.78125 19.062321 \r\nL 48.12125 20.043289 \r\nL 52.46125 20.518128 \r\nL 56.80125 27.082629 \r\nL 61.14125 38.080132 \r\nL 65.48125 47.244035 \r\nL 69.82125 90.07116 \r\nL 74.16125 90.82213 \r\nL 78.50125 92.383257 \r\nL 82.84125 93.5572 \r\nL 87.18125 94.629524 \r\nL 91.52125 101.956598 \r\nL 95.86125 102.670847 \r\nL 100.20125 103.026019 \r\nL 104.54125 103.93087 \r\nL 108.88125 104.564671 \r\nL 113.22125 108.245685 \r\nL 117.56125 108.957584 \r\nL 121.90125 109.349083 \r\nL 126.24125 109.367547 \r\nL 130.58125 109.719548 \r\nL 134.92125 111.404785 \r\nL 139.26125 111.756305 \r\nL 143.60125 112.169154 \r\nL 147.94125 112.499389 \r\nL 152.28125 112.915446 \r\nL 156.62125 114.903217 \r\nL 160.96125 115.127225 \r\nL 165.30125 114.983588 \r\nL 169.64125 115.30367 \r\nL 173.98125 115.401143 \r\nL 178.32125 117.148262 \r\nL 182.66125 117.159426 \r\nL 187.00125 117.347615 \r\nL 191.34125 117.379144 \r\nL 195.68125 117.707257 \r\nL 200.02125 118.670007 \r\nL 204.36125 118.855509 \r\nL 208.70125 118.900946 \r\nL 213.04125 119.097948 \r\nL 217.38125 119.130388 \r\nL 221.72125 120.22226 \r\nL 226.06125 119.984933 \r\nL 230.40125 120.117081 \r\nL 234.74125 120.151792 \r\nL 239.08125 120.365466 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_22\">\r\n    <path clip-path=\"url(#pc51e788df5)\" d=\"M 26.42125 140.302442 \r\nL 30.76125 140.422727 \r\nL 35.10125 140.373554 \r\nL 39.44125 140.294499 \r\nL 43.78125 140.261508 \r\nL 48.12125 140.007403 \r\nL 52.46125 139.038314 \r\nL 56.80125 136.500224 \r\nL 61.14125 132.722213 \r\nL 65.48125 129.35146 \r\nL 69.82125 113.685399 \r\nL 74.16125 113.158868 \r\nL 78.50125 112.449262 \r\nL 82.84125 112.019564 \r\nL 87.18125 111.560962 \r\nL 91.52125 108.442786 \r\nL 95.86125 108.127321 \r\nL 100.20125 107.922307 \r\nL 104.54125 107.547456 \r\nL 108.88125 107.307445 \r\nL 113.22125 106.05524 \r\nL 117.56125 105.653534 \r\nL 121.90125 105.447007 \r\nL 126.24125 105.399346 \r\nL 130.58125 105.264883 \r\nL 134.92125 104.70714 \r\nL 139.26125 104.400753 \r\nL 143.60125 104.347041 \r\nL 147.94125 104.157914 \r\nL 152.28125 103.977814 \r\nL 156.62125 102.741727 \r\nL 160.96125 102.925559 \r\nL 165.30125 102.942959 \r\nL 169.64125 102.885842 \r\nL 173.98125 102.852766 \r\nL 178.32125 102.083564 \r\nL 182.66125 102.076755 \r\nL 187.00125 101.98673 \r\nL 191.34125 101.896327 \r\nL 195.68125 101.738641 \r\nL 200.02125 101.502564 \r\nL 204.36125 101.323271 \r\nL 208.70125 101.286202 \r\nL 213.04125 101.20639 \r\nL 217.38125 101.219809 \r\nL 221.72125 100.726385 \r\nL 226.06125 100.792201 \r\nL 230.40125 100.706716 \r\nL 234.74125 100.667377 \r\nL 239.08125 100.602671 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_23\">\r\n    <path clip-path=\"url(#pc51e788df5)\" d=\"M 43.78125 140.292456 \r\nL 65.48125 115.770788 \r\nL 87.18125 110.598846 \r\nL 108.88125 106.262498 \r\nL 130.58125 105.437827 \r\nL 152.28125 103.810331 \r\nL 173.98125 103.624643 \r\nL 195.68125 102.029916 \r\nL 217.38125 105.754588 \r\nL 239.08125 100.533493 \r\n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 43.78125 146.6 \r\nL 43.78125 10.7 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 239.08125 146.6 \r\nL 239.08125 10.7 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 43.78125 146.6 \r\nL 239.08125 146.6 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 43.78125 10.7 \r\nL 239.08125 10.7 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"legend_1\">\r\n    <g id=\"patch_7\">\r\n     <path d=\"M 154.3125 62.734375 \r\nL 232.08125 62.734375 \r\nQ 234.08125 62.734375 234.08125 60.734375 \r\nL 234.08125 17.7 \r\nQ 234.08125 15.7 232.08125 15.7 \r\nL 154.3125 15.7 \r\nQ 152.3125 15.7 152.3125 17.7 \r\nL 152.3125 60.734375 \r\nQ 152.3125 62.734375 154.3125 62.734375 \r\nz\r\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\r\n    </g>\r\n    <g id=\"line2d_24\">\r\n     <path d=\"M 156.3125 23.798437 \r\nL 176.3125 23.798437 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_25\"/>\r\n    <g id=\"text_13\">\r\n     <!-- train loss -->\r\n     <defs>\r\n      <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-74\"/>\r\n      <path d=\"M 41.109375 46.296875 \r\nQ 39.59375 47.171875 37.8125 47.578125 \r\nQ 36.03125 48 33.890625 48 \r\nQ 26.265625 48 22.1875 43.046875 \r\nQ 18.109375 38.09375 18.109375 28.8125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 20.953125 51.171875 25.484375 53.578125 \r\nQ 30.03125 56 36.53125 56 \r\nQ 37.453125 56 38.578125 55.875 \r\nQ 39.703125 55.765625 41.0625 55.515625 \r\nz\r\n\" id=\"DejaVuSans-72\"/>\r\n      <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-61\"/>\r\n      <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-69\"/>\r\n      <path id=\"DejaVuSans-20\"/>\r\n      <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-6c\"/>\r\n      <path d=\"M 44.28125 53.078125 \r\nL 44.28125 44.578125 \r\nQ 40.484375 46.53125 36.375 47.5 \r\nQ 32.28125 48.484375 27.875 48.484375 \r\nQ 21.1875 48.484375 17.84375 46.4375 \r\nQ 14.5 44.390625 14.5 40.28125 \r\nQ 14.5 37.15625 16.890625 35.375 \r\nQ 19.28125 33.59375 26.515625 31.984375 \r\nL 29.59375 31.296875 \r\nQ 39.15625 29.25 43.1875 25.515625 \r\nQ 47.21875 21.78125 47.21875 15.09375 \r\nQ 47.21875 7.46875 41.1875 3.015625 \r\nQ 35.15625 -1.421875 24.609375 -1.421875 \r\nQ 20.21875 -1.421875 15.453125 -0.5625 \r\nQ 10.6875 0.296875 5.421875 2 \r\nL 5.421875 11.28125 \r\nQ 10.40625 8.6875 15.234375 7.390625 \r\nQ 20.0625 6.109375 24.8125 6.109375 \r\nQ 31.15625 6.109375 34.5625 8.28125 \r\nQ 37.984375 10.453125 37.984375 14.40625 \r\nQ 37.984375 18.0625 35.515625 20.015625 \r\nQ 33.0625 21.96875 24.703125 23.78125 \r\nL 21.578125 24.515625 \r\nQ 13.234375 26.265625 9.515625 29.90625 \r\nQ 5.8125 33.546875 5.8125 39.890625 \r\nQ 5.8125 47.609375 11.28125 51.796875 \r\nQ 16.75 56 26.8125 56 \r\nQ 31.78125 56 36.171875 55.265625 \r\nQ 40.578125 54.546875 44.28125 53.078125 \r\nz\r\n\" id=\"DejaVuSans-73\"/>\r\n     </defs>\r\n     <g transform=\"translate(184.3125 27.298437)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-74\"/>\r\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-72\"/>\r\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-69\"/>\r\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-6e\"/>\r\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-20\"/>\r\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-6c\"/>\r\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-6f\"/>\r\n      <use x=\"353.515625\" xlink:href=\"#DejaVuSans-73\"/>\r\n      <use x=\"405.615234\" xlink:href=\"#DejaVuSans-73\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_26\">\r\n     <path d=\"M 156.3125 38.476562 \r\nL 176.3125 38.476562 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_27\"/>\r\n    <g id=\"text_14\">\r\n     <!-- train acc -->\r\n     <g transform=\"translate(184.3125 41.976562)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-74\"/>\r\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-72\"/>\r\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-69\"/>\r\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-6e\"/>\r\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-20\"/>\r\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"325.830078\" xlink:href=\"#DejaVuSans-63\"/>\r\n      <use x=\"380.810547\" xlink:href=\"#DejaVuSans-63\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_28\">\r\n     <path d=\"M 156.3125 53.154687 \r\nL 176.3125 53.154687 \r\n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_29\"/>\r\n    <g id=\"text_15\">\r\n     <!-- test acc -->\r\n     <g transform=\"translate(184.3125 56.654687)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-74\"/>\r\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-65\"/>\r\n      <use x=\"100.732422\" xlink:href=\"#DejaVuSans-73\"/>\r\n      <use x=\"152.832031\" xlink:href=\"#DejaVuSans-74\"/>\r\n      <use x=\"192.041016\" xlink:href=\"#DejaVuSans-20\"/>\r\n      <use x=\"223.828125\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"285.107422\" xlink:href=\"#DejaVuSans-63\"/>\r\n      <use x=\"340.087891\" xlink:href=\"#DejaVuSans-63\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"pc51e788df5\">\r\n   <rect height=\"135.9\" width=\"195.3\" x=\"43.78125\" y=\"10.7\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```



### 3.2.2 AlexNet
AlexNet是我们真正意义上的一个深度卷积神经网络，其发表在2012年的论文 [ImageNet
Classification with Deep Convolutional Neural Networks](chrome-
extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-
viewer/web/viewer.html?file=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F2012%2Ffile%2Fc399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
是深度学习的奠基作之一，其也赢下了当年ImageNet数据集的冠军（1百万个样本，1000种类别）。截止到我写这篇博客(2022年)为知，其引用数已经达到了102838次。论文当中，有些技术细节现在来看或许是多余的，但是其很多工作，对后面的一系列研究起到重要的启发式作用。

## 学习表征
在合理地复杂性前提下，特征应该由多个共同学习的神经网络层组成，每个层都有可学习的参数。
在计算机视觉当中，最底层可能检测边缘，纹理等信息。正如AlexNet论文当中描述的，在底层的网络当中，模型学到了一些类似于传统滤波器的特征抽取器，论文中的图像生动表示了这种底层特征信息，其为AlexNet第一层学习到的特征抽取器:
![](https://zh-v2.d2l.ai/_images/filters.png)
AlexNet的更高层建立在这些底层表示的基础上，以表示更大的特征，如眼睛、鼻子、草叶等等。而更高的层可以检测整个物体，如人、飞机、狗或飞盘。（高层具有更多的**语义信息**）最终的隐藏神经元可以学习图像的综合表示，从而使属于不同类别的数据易于区分。尽管一直有一群执着的研究者不断钻研，试图学习视觉数据的逐级表征，然而很长一段时间里这些尝试都未有突破。深度卷积神经网络的突破出现在2012年。突破可归因于两个关键因素:
数据与硬件计算。

AlexNet论文当中采用的硬件是两块3GB的GTX
580实现卷积运算。但实际上，在今天的硬件水平下，AlexNet做模型并行是有些多余的。但面对当今更大的模型(eg BERT,
Transformer)，模型并行又重新流行了起来。

### 网络架构
2012年，AlexNet横空出世。它首次证明了学习到的特征可以超越手工设计的特征。它一举打破了计算机视觉研究的现状。
AlexNet使用了**8层**卷积神经网络，并以很大的优势赢得了2012年ImageNet图像识别挑战赛。下图为其论文中的网络示意图:
![](https://s2.loli.net/2022/02/06/qWFM1EjodptX7ay.png)
为了让读者更容易看懂，我们简化其分开GPU进行训练的设计，其架构如下图所示（左图为LeNet，右图为AlexNet）:
![](https://zh-v2.d2l.ai/_images/alexnet.svg)

**注意**:后续的代码实现亦不会考虑分开GPU来进行训练。
可以看到，AlexNet和LeNet的设计理念非常相似，但也存在显著差异:
* AlexNet比相对较小的LeNet5要深得多。
*
AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
* AlexNet使用ReLU而不是sigmoid作为其激活函数。
*
AlexNet十大池化层（LeNet使用平均池化层），并且卷积层有更大的和窗口和步长（因为图片有更大的像素）。
![](https://s2.loli.net/2022/02/07/ehTzgiSsrUdY2Nu.png)
主要的改进为:
* dropout
丢弃法，而不是权重衰减，避免过拟合。（当然，论文中还用到了两种数据增强的方法避免过拟合，随机crop与通道变换）
    *
后面的论文基本解释dropout等价于$l2$正则，但无法用表达式写出来。
* ReLU 取代 sigmoid，更简单更快
* 使用最大池化层
MaxPooling

AlexNet的出现极大的推动了计算机视觉，和有监督学习的发展。

```{.python .input  n=3}
import torch
from torch import nn
from d2l import torch as d2l
```

下面通过Pytorch API构造AlexNet。这里为了捕捉对象，我们开始使用一个核大小为11的窗口,我们使用MNIST数据集，因此输入通道数为1。

```{.python .input  n=4}
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)
```

测试一下每层的输出:

```{.python .input  n=1}
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.json .output n=1}
[
 {
  "ename": "NameError",
  "evalue": "name 'torch' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[1;32m<ipython-input-1-e0e9fffea044>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrandn\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m224\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m224\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mlayer\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mnet\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m     \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlayer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlayer\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__class__\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__name__\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'output shape:\\t'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
   "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
  ]
 }
]
```

#### 训练网络
以Fashion-MNIST图像为数据集，其分辨率为$28 \times 28$像素，远低于ImageNet的$224 \times
224$ 为了令其训练成功，我们将其增加到$224 \times 224$（**但实际应用当中，并不建议这样做！**）
为了演示方便，我们借助d2l库的函数完成数据集导入和训练。

* 读取数据集

```{.python .input  n=6}
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "b33dde20833e4eb7a505136152aac383",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/26421880 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ../data/FashionMNIST/raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "f3eaa3734da44412acebf93b92484121",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/29515 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "2c34e8bc3e8346b98eecb1b9012097e7",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/4422102 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ../data/FashionMNIST/raw\n\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz\nDownloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "90f1e3b247324b518c4ca4bfea4e8e73",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "  0%|          | 0/5148 [00:00<?, ?it/s]"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Extracting ../data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ../data/FashionMNIST/raw\n\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.\n  cpuset_checked))\n"
 }
]
```

* 训练网络

下面函数支持在GPU和CPU上训练

```{.python .input  n=7}
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """计算模型精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估状态
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量, 预测总数
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调中所使用的，这里不用理会
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input  n=8}
def train(net, train_iter, test_iter, num_epochs, lr, device):
    """模型训练函数"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input  n=20}
lr, num_epochs = 0.01, 10
train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.327, train acc 0.880, test acc 0.880\n384.6 examples/sec on cuda:0\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 238.965625 180.65625\" width=\"238.965625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m8999a6ccfb\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"51.803125\" xlink:href=\"#m8999a6ccfb\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"95.203125\" xlink:href=\"#m8999a6ccfb\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"138.603125\" xlink:href=\"#m8999a6ccfb\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"182.003125\" xlink:href=\"#m8999a6ccfb\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m8999a6ccfb\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n     </defs>\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 30.103125 116.899359 \nL 225.403125 116.899359 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mf2fe10df9b\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf2fe10df9b\" y=\"116.899359\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.5 -->\n      <defs>\n       <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       <path d=\"M 10.796875 72.90625 \nL 49.515625 72.90625 \nL 49.515625 64.59375 \nL 19.828125 64.59375 \nL 19.828125 46.734375 \nQ 21.96875 47.46875 24.109375 47.828125 \nQ 26.265625 48.1875 28.421875 48.1875 \nQ 40.625 48.1875 47.75 41.5 \nQ 54.890625 34.8125 54.890625 23.390625 \nQ 54.890625 11.625 47.5625 5.09375 \nQ 40.234375 -1.421875 26.90625 -1.421875 \nQ 22.3125 -1.421875 17.546875 -0.640625 \nQ 12.796875 0.140625 7.71875 1.703125 \nL 7.71875 11.625 \nQ 12.109375 9.234375 16.796875 8.0625 \nQ 21.484375 6.890625 26.703125 6.890625 \nQ 35.15625 6.890625 40.078125 11.328125 \nQ 45.015625 15.765625 45.015625 23.390625 \nQ 45.015625 31 40.078125 35.4375 \nQ 35.15625 39.890625 26.703125 39.890625 \nQ 22.75 39.890625 18.8125 39.015625 \nQ 14.890625 38.140625 10.796875 36.28125 \nz\n\" id=\"DejaVuSans-53\"/>\n      </defs>\n      <g transform=\"translate(7.2 120.698578)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 30.103125 87.986431 \nL 225.403125 87.986431 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf2fe10df9b\" y=\"87.986431\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 91.78565)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 30.103125 59.073503 \nL 225.403125 59.073503 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf2fe10df9b\" y=\"59.073503\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 1.5 -->\n      <g transform=\"translate(7.2 62.872722)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path clip-path=\"url(#p2c9c077327)\" d=\"M 30.103125 30.160575 \nL 225.403125 30.160575 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf2fe10df9b\" y=\"30.160575\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 2.0 -->\n      <g transform=\"translate(7.2 33.959794)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#p2c9c077327)\" d=\"M 12.70611 13.377273 \nL 17.009095 24.856775 \nL 21.31208 44.043224 \nL 25.615065 56.392495 \nL 29.91805 65.197866 \nL 30.103125 65.524453 \nL 34.40611 102.225582 \nL 38.709095 104.203542 \nL 43.01208 105.862436 \nL 47.315065 106.945534 \nL 51.61805 107.875138 \nL 51.803125 107.895927 \nL 56.10611 113.428476 \nL 60.409095 113.249841 \nL 64.71208 113.805135 \nL 69.015065 114.261803 \nL 73.31805 114.826614 \nL 73.503125 114.861679 \nL 77.80611 117.934048 \nL 82.109095 118.073277 \nL 86.41208 118.224545 \nL 90.715065 118.542155 \nL 95.01805 118.531184 \nL 95.203125 118.537016 \nL 99.50611 119.921282 \nL 103.809095 120.598708 \nL 108.11208 120.738918 \nL 112.415065 121.033729 \nL 116.71805 121.102852 \nL 116.903125 121.111692 \nL 121.20611 122.393399 \nL 125.509095 122.258144 \nL 129.81208 122.394312 \nL 134.115065 122.549322 \nL 138.41805 122.734672 \nL 138.603125 122.726054 \nL 142.90611 124.254123 \nL 147.209095 124.301077 \nL 151.51208 124.106378 \nL 155.815065 124.056031 \nL 160.11805 124.136484 \nL 160.303125 124.166699 \nL 164.60611 125.129659 \nL 168.909095 125.029777 \nL 173.21208 125.148875 \nL 177.515065 125.103546 \nL 181.81805 125.197247 \nL 182.003125 125.211834 \nL 186.30611 126.305746 \nL 190.609095 125.858529 \nL 194.91208 125.988027 \nL 199.215065 126.100054 \nL 203.51805 126.117479 \nL 203.703125 126.108091 \nL 208.00611 126.905588 \nL 212.309095 126.695521 \nL 216.61208 126.880274 \nL 220.915065 126.787336 \nL 225.21805 126.882269 \nL 225.403125 126.900254 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path clip-path=\"url(#p2c9c077327)\" d=\"M 12.70611 136.922727 \nL 17.009095 131.639997 \nL 21.31208 125.081316 \nL 25.615065 120.830439 \nL 29.91805 117.678531 \nL 30.103125 117.560501 \nL 34.40611 104.327676 \nL 38.709095 103.730181 \nL 43.01208 103.048486 \nL 47.315065 102.57891 \nL 51.61805 102.249559 \nL 51.803125 102.238577 \nL 56.10611 100.339518 \nL 60.409095 100.186501 \nL 64.71208 99.986527 \nL 69.015065 99.750524 \nL 73.31805 99.535086 \nL 73.503125 99.521725 \nL 77.80611 98.367299 \nL 82.109095 98.245857 \nL 86.41208 98.15518 \nL 90.715065 98.045478 \nL 95.01805 98.062237 \nL 95.203125 98.060659 \nL 99.50611 97.454055 \nL 103.809095 97.237888 \nL 108.11208 97.282417 \nL 112.415065 97.178381 \nL 116.71805 97.150935 \nL 116.903125 97.146047 \nL 121.20611 96.715687 \nL 125.509095 96.744833 \nL 129.81208 96.587768 \nL 134.115065 96.548097 \nL 138.41805 96.50292 \nL 138.603125 96.51189 \nL 142.90611 96.021038 \nL 147.209095 95.950602 \nL 151.51208 95.982177 \nL 155.815065 96.001608 \nL 160.11805 95.938458 \nL 160.303125 95.924957 \nL 164.60611 95.797585 \nL 168.909095 95.770868 \nL 173.21208 95.668047 \nL 177.515065 95.65064 \nL 181.81805 95.581904 \nL 182.003125 95.570292 \nL 186.30611 95.030071 \nL 190.609095 95.234094 \nL 194.91208 95.230855 \nL 199.215065 95.221949 \nL 203.51805 95.243809 \nL 203.703125 95.245503 \nL 208.00611 94.913487 \nL 212.309095 94.99121 \nL 216.61208 94.903771 \nL 220.915065 94.940204 \nL 225.21805 94.9028 \nL 225.403125 94.898548 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path clip-path=\"url(#p2c9c077327)\" d=\"M 30.103125 103.113675 \nL 51.803125 99.673036 \nL 73.503125 98.302564 \nL 95.203125 97.250133 \nL 116.903125 97.140264 \nL 138.603125 96.631396 \nL 160.303125 95.567401 \nL 182.003125 96.024225 \nL 203.703125 95.301402 \nL 225.403125 94.902403 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 140.634375 59.234375 \nL 218.403125 59.234375 \nQ 220.403125 59.234375 220.403125 57.234375 \nL 220.403125 14.2 \nQ 220.403125 12.2 218.403125 12.2 \nL 140.634375 12.2 \nQ 138.634375 12.2 138.634375 14.2 \nL 138.634375 57.234375 \nQ 138.634375 59.234375 140.634375 59.234375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 142.634375 20.298438 \nL 162.634375 20.298438 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_23\"/>\n    <g id=\"text_11\">\n     <!-- train loss -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n      <path id=\"DejaVuSans-32\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n     </defs>\n     <g transform=\"translate(170.634375 23.798438)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"353.515625\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"405.615234\" xlink:href=\"#DejaVuSans-115\"/>\n     </g>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 142.634375 34.976562 \nL 162.634375 34.976562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_25\"/>\n    <g id=\"text_12\">\n     <!-- train acc -->\n     <g transform=\"translate(170.634375 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"325.830078\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"380.810547\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 142.634375 49.654688 \nL 162.634375 49.654688 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_27\"/>\n    <g id=\"text_13\">\n     <!-- test acc -->\n     <g transform=\"translate(170.634375 53.154688)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"100.732422\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"152.832031\" xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"192.041016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"223.828125\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"285.107422\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"340.087891\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p2c9c077327\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"30.103125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

### 3.2.3 深度神经网络设计
在AlexNet当中，其设计不太规则，让人感觉比较随意，没有提供一个通用的模板来指导后续的研究人员设计新的网络。
后面将介绍一些常用于设计深层神经网络的启发式概念。

#### VGG 使用块的网络
与芯片设计中工程师从放置晶体管到逻辑元件再到逻辑块的过程类似，神经网络架构的设计也逐渐变得更加抽象。研究人员开始从单个神经元的角度思考问题，发展到整个层，现在又转向块，重复层的模式。

##### VGG块
经典卷积神经网络的基本组成部分是下面的这个序列：
 1. 带填充以保持分辨率的卷积层。
 2. 非线性激活函数，如ReLU。
 3.
汇聚层，如最大汇聚层。

而对VGG块来说:
如何训练更深更大网路：
* 选项
  * 更多的全连接层（太贵）
  * 更多的卷积层
  *
**将卷积层组合成块**(VGG核心思想)

一个VGG块由一系列**卷积层**组成，后面再加上用于空间下采样的**最大汇聚层**。

该函数有三个参数，分别对应于卷积层的数量`num_convs`、输入通道的数量`in_channels` 和输出通道的数量`out_channels`。
我们下面[原始论文](https://arxiv.org/abs/1409.1556)中的网络结构:

```{.python .input  n=3}
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

##### VGG网络
VGG网络可以分为两部分：
* 第一部分主要由**卷积层**和**汇聚层**组成
* 第二部分由**全连接层**组成
![](https://zh-v2.d2l.ai/_images/vgg.svg)
其中有超参数变量conv_arch。该变量指定了每个VGG块里卷积层个数和输出通道数。
原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。
第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。

```{.python .input  n=4}
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

下面的代码实现了VGG-11。可以通过在conv_arch上执行for循环来简单实现。

```{.python .input  n=5}
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

下面构建一个高度和宽度为224的单通道数据样本，以观察每个层输出的形状。

```{.python .input  n=6}
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential output shape:\t torch.Size([1, 64, 112, 112])\nSequential output shape:\t torch.Size([1, 128, 56, 56])\nSequential output shape:\t torch.Size([1, 256, 28, 28])\nSequential output shape:\t torch.Size([1, 512, 14, 14])\nSequential output shape:\t torch.Size([1, 512, 7, 7])\nFlatten output shape:\t torch.Size([1, 25088])\nLinear output shape:\t torch.Size([1, 4096])\nReLU output shape:\t torch.Size([1, 4096])\nDropout output shape:\t torch.Size([1, 4096])\nLinear output shape:\t torch.Size([1, 4096])\nReLU output shape:\t torch.Size([1, 4096])\nDropout output shape:\t torch.Size([1, 4096])\nLinear output shape:\t torch.Size([1, 10])\n"
 }
]
```

#### 训练模型
由于VGG-11比AlexNet计算量更大，因此构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。

```{.python .input  n=7}
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input  n=10}
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.172, train acc 0.936, test acc 0.923\n366.9 examples/sec on cuda:0\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 238.965625 180.65625\" width=\"238.965625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m3908360041\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"51.803125\" xlink:href=\"#m3908360041\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"95.203125\" xlink:href=\"#m3908360041\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"138.603125\" xlink:href=\"#m3908360041\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"182.003125\" xlink:href=\"#m3908360041\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m3908360041\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n     </defs>\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 30.103125 116.598969 \nL 225.403125 116.598969 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mf220daace3\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf220daace3\" y=\"116.598969\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.5 -->\n      <defs>\n       <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       <path d=\"M 10.796875 72.90625 \nL 49.515625 72.90625 \nL 49.515625 64.59375 \nL 19.828125 64.59375 \nL 19.828125 46.734375 \nQ 21.96875 47.46875 24.109375 47.828125 \nQ 26.265625 48.1875 28.421875 48.1875 \nQ 40.625 48.1875 47.75 41.5 \nQ 54.890625 34.8125 54.890625 23.390625 \nQ 54.890625 11.625 47.5625 5.09375 \nQ 40.234375 -1.421875 26.90625 -1.421875 \nQ 22.3125 -1.421875 17.546875 -0.640625 \nQ 12.796875 0.140625 7.71875 1.703125 \nL 7.71875 11.625 \nQ 12.109375 9.234375 16.796875 8.0625 \nQ 21.484375 6.890625 26.703125 6.890625 \nQ 35.15625 6.890625 40.078125 11.328125 \nQ 45.015625 15.765625 45.015625 23.390625 \nQ 45.015625 31 40.078125 35.4375 \nQ 35.15625 39.890625 26.703125 39.890625 \nQ 22.75 39.890625 18.8125 39.015625 \nQ 14.890625 38.140625 10.796875 36.28125 \nz\n\" id=\"DejaVuSans-53\"/>\n      </defs>\n      <g transform=\"translate(7.2 120.398187)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 30.103125 85.980631 \nL 225.403125 85.980631 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf220daace3\" y=\"85.980631\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 89.77985)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 30.103125 55.362293 \nL 225.403125 55.362293 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf220daace3\" y=\"55.362293\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 1.5 -->\n      <g transform=\"translate(7.2 59.161512)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path clip-path=\"url(#p681ca7cf25)\" d=\"M 30.103125 24.743955 \nL 225.403125 24.743955 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf220daace3\" y=\"24.743955\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 2.0 -->\n      <g transform=\"translate(7.2 28.543174)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#p681ca7cf25)\" d=\"M 12.70611 13.377273 \nL 17.009095 50.147677 \nL 21.31208 68.802595 \nL 25.615065 80.030423 \nL 29.91805 87.523573 \nL 30.103125 87.737185 \nL 34.40611 120.964589 \nL 38.709095 121.177221 \nL 43.01208 121.731382 \nL 47.315065 122.375279 \nL 51.61805 122.729494 \nL 51.803125 122.781722 \nL 56.10611 126.735 \nL 60.409095 126.289187 \nL 64.71208 126.591965 \nL 69.015065 127.105413 \nL 73.31805 127.305672 \nL 73.503125 127.317991 \nL 77.80611 129.580226 \nL 82.109095 129.656632 \nL 86.41208 129.789098 \nL 90.715065 129.818458 \nL 95.01805 129.904221 \nL 95.203125 129.901756 \nL 99.50611 131.452853 \nL 103.809095 131.211719 \nL 108.11208 131.16703 \nL 112.415065 131.449112 \nL 116.71805 131.510811 \nL 116.903125 131.533934 \nL 121.20611 133.233343 \nL 125.509095 133.07257 \nL 129.81208 132.766162 \nL 134.115065 132.891421 \nL 138.41805 132.892508 \nL 138.603125 132.881948 \nL 142.90611 134.005514 \nL 147.209095 134.023289 \nL 151.51208 133.796611 \nL 155.815065 133.870735 \nL 160.11805 133.983154 \nL 160.303125 133.977239 \nL 164.60611 134.903159 \nL 168.909095 134.969859 \nL 173.21208 134.994292 \nL 177.515065 134.983474 \nL 181.81805 134.944605 \nL 182.003125 134.946881 \nL 186.30611 135.64866 \nL 190.609095 135.656199 \nL 194.91208 135.812891 \nL 199.215065 135.976115 \nL 203.51805 135.836169 \nL 203.703125 135.859977 \nL 208.00611 136.922727 \nL 212.309095 136.796316 \nL 216.61208 136.748626 \nL 220.915065 136.65472 \nL 225.21805 136.689111 \nL 225.403125 136.676066 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path clip-path=\"url(#p681ca7cf25)\" d=\"M 12.70611 134.526541 \nL 17.009095 121.761183 \nL 21.31208 114.99226 \nL 25.615065 110.841312 \nL 29.91805 108.066782 \nL 30.103125 107.98603 \nL 34.40611 95.723765 \nL 38.709095 95.546289 \nL 43.01208 95.319087 \nL 47.315065 95.177192 \nL 51.61805 95.02621 \nL 51.803125 95.005896 \nL 56.10611 93.558052 \nL 60.409095 93.804974 \nL 64.71208 93.650648 \nL 69.015065 93.423017 \nL 73.31805 93.329649 \nL 73.503125 93.329032 \nL 77.80611 92.539499 \nL 82.109095 92.392889 \nL 86.41208 92.361166 \nL 90.715065 92.3826 \nL 95.01805 92.346076 \nL 95.203125 92.34108 \nL 99.50611 91.7833 \nL 103.809095 91.809021 \nL 108.11208 91.810736 \nL 112.415065 91.739574 \nL 116.71805 91.742146 \nL 116.903125 91.736878 \nL 121.20611 91.202004 \nL 125.509095 91.225153 \nL 129.81208 91.354616 \nL 134.115065 91.250874 \nL 138.41805 91.249331 \nL 138.603125 91.259232 \nL 142.90611 90.949938 \nL 147.209095 90.877919 \nL 151.51208 90.970515 \nL 155.815065 90.967942 \nL 160.11805 90.908784 \nL 160.303125 90.912224 \nL 164.60611 90.430373 \nL 168.909095 90.371214 \nL 173.21208 90.399507 \nL 177.515065 90.444519 \nL 181.81805 90.470497 \nL 182.003125 90.461114 \nL 186.30611 90.229748 \nL 190.609095 90.276046 \nL 194.91208 90.224604 \nL 199.215065 90.157729 \nL 203.51805 90.187566 \nL 203.703125 90.178405 \nL 208.00611 89.869654 \nL 212.309095 89.892803 \nL 216.61208 89.888516 \nL 220.915065 89.910807 \nL 225.21805 89.904634 \nL 225.403125 89.908964 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path clip-path=\"url(#p681ca7cf25)\" d=\"M 30.103125 96.109177 \nL 51.803125 93.512742 \nL 73.503125 93.476 \nL 95.203125 92.355369 \nL 116.903125 92.061433 \nL 138.603125 91.70626 \nL 160.303125 91.197996 \nL 182.003125 91.075522 \nL 203.703125 90.793834 \nL 225.403125 90.708102 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 140.634375 59.234375 \nL 218.403125 59.234375 \nQ 220.403125 59.234375 220.403125 57.234375 \nL 220.403125 14.2 \nQ 220.403125 12.2 218.403125 12.2 \nL 140.634375 12.2 \nQ 138.634375 12.2 138.634375 14.2 \nL 138.634375 57.234375 \nQ 138.634375 59.234375 140.634375 59.234375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 142.634375 20.298438 \nL 162.634375 20.298438 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_23\"/>\n    <g id=\"text_11\">\n     <!-- train loss -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n      <path id=\"DejaVuSans-32\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n     </defs>\n     <g transform=\"translate(170.634375 23.798438)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"353.515625\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"405.615234\" xlink:href=\"#DejaVuSans-115\"/>\n     </g>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 142.634375 34.976562 \nL 162.634375 34.976562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_25\"/>\n    <g id=\"text_12\">\n     <!-- train acc -->\n     <g transform=\"translate(170.634375 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"325.830078\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"380.810547\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 142.634375 49.654688 \nL 162.634375 49.654688 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_27\"/>\n    <g id=\"text_13\">\n     <!-- test acc -->\n     <g transform=\"translate(170.634375 53.154688)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"100.732422\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"152.832031\" xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"192.041016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"223.828125\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"285.107422\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"340.087891\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p681ca7cf25\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"30.103125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

### 3.2.4 NiN块 网络中的网络
`LeNet`、`AlexNet`和`VGG`都有一个共同的设计模式：通过一系列的卷积层与汇聚层来提取**空间结构特征**,然后通过**全连接层**对特征的表征进行处理。AlexNet和VGG对LeNet的改进主要在于如何构造一个更深的网络。
然而，如果使用了全连接层，可能会**完全放弃表征**的空间结构，会带来*过拟合*，内存也要求很大。
* 卷积层需要较少的参数

$$
c_i \times
c_o \times k^2
$$

* 但卷积后的第一个全连接层的参数：
  * LeNet
    * 16x5x5x120= 48k
  *
AlexNet
    * 256x5x5x4096 = 26M
  * VGG
    * 512x7x7x4096= 102M

**NiN的设计思想**:
完全不要全连接层, 使用卷积层来进行替代。在每个像素的通道上分别使用多层感知机。 实际上NiN的使用并不算广泛，但其提出了新的概念。
![](https://zh-v2.d2l.ai/_images/nin.svg)

* 1 $\times$ 1的卷积层（**等价于一个全连接层**）
*
交替使用**NiN块**和步幅为2的**最大池化层**（高宽减半）
  * 逐步减小高宽和增大通道数
* 最后使用全局平均池化层得到输出
  *
其（最后一层）输入通道数是类别数（每个通道拿出一个值）

```{.python .input  n=2}
import torch
from torch import nn
from d2l import torch as d2l
```

#### NiN块代码实现

```{.python .input  n=3}
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
```

#### NiN模型代码实现

```{.python .input  n=4}
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),  # 10个类别
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化层，池化后的每个通道上的大小是一个1x1
    nn.Flatten()  # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
)
```

```{.python .input  n=5}
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential output shape:\t torch.Size([1, 96, 54, 54])\nMaxPool2d output shape:\t torch.Size([1, 96, 26, 26])\nSequential output shape:\t torch.Size([1, 256, 26, 26])\nMaxPool2d output shape:\t torch.Size([1, 256, 12, 12])\nSequential output shape:\t torch.Size([1, 384, 12, 12])\nMaxPool2d output shape:\t torch.Size([1, 384, 5, 5])\nDropout output shape:\t torch.Size([1, 384, 5, 5])\nSequential output shape:\t torch.Size([1, 10, 5, 5])\nAdaptiveAvgPool2d output shape:\t torch.Size([1, 10, 1, 1])\nFlatten output shape:\t torch.Size([1, 10])\n"
 }
]
```

#### 训练

```{.python .input  n=6}
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.349, train acc 0.869, test acc 0.863\n379.2 examples/sec on cuda:0\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 238.965625 180.65625\" width=\"238.965625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m9dc29fba55\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"51.803125\" xlink:href=\"#m9dc29fba55\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"95.203125\" xlink:href=\"#m9dc29fba55\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"138.603125\" xlink:href=\"#m9dc29fba55\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"182.003125\" xlink:href=\"#m9dc29fba55\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m9dc29fba55\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n     </defs>\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 30.103125 115.543983 \nL 225.403125 115.543983 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mf3c32b987b\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf3c32b987b\" y=\"115.543983\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.5 -->\n      <defs>\n       <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       <path d=\"M 10.796875 72.90625 \nL 49.515625 72.90625 \nL 49.515625 64.59375 \nL 19.828125 64.59375 \nL 19.828125 46.734375 \nQ 21.96875 47.46875 24.109375 47.828125 \nQ 26.265625 48.1875 28.421875 48.1875 \nQ 40.625 48.1875 47.75 41.5 \nQ 54.890625 34.8125 54.890625 23.390625 \nQ 54.890625 11.625 47.5625 5.09375 \nQ 40.234375 -1.421875 26.90625 -1.421875 \nQ 22.3125 -1.421875 17.546875 -0.640625 \nQ 12.796875 0.140625 7.71875 1.703125 \nL 7.71875 11.625 \nQ 12.109375 9.234375 16.796875 8.0625 \nQ 21.484375 6.890625 26.703125 6.890625 \nQ 35.15625 6.890625 40.078125 11.328125 \nQ 45.015625 15.765625 45.015625 23.390625 \nQ 45.015625 31 40.078125 35.4375 \nQ 35.15625 39.890625 26.703125 39.890625 \nQ 22.75 39.890625 18.8125 39.015625 \nQ 14.890625 38.140625 10.796875 36.28125 \nz\n\" id=\"DejaVuSans-53\"/>\n      </defs>\n      <g transform=\"translate(7.2 119.343201)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 30.103125 87.185104 \nL 225.403125 87.185104 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf3c32b987b\" y=\"87.185104\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 90.984323)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 30.103125 58.826226 \nL 225.403125 58.826226 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf3c32b987b\" y=\"58.826226\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 1.5 -->\n      <g transform=\"translate(7.2 62.625445)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path clip-path=\"url(#p18a32d5f07)\" d=\"M 30.103125 30.467347 \nL 225.403125 30.467347 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#mf3c32b987b\" y=\"30.467347\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 2.0 -->\n      <g transform=\"translate(7.2 34.266566)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#p18a32d5f07)\" d=\"M 12.70611 13.377273 \nL 17.009095 14.470888 \nL 21.31208 16.100973 \nL 25.615065 19.652231 \nL 29.91805 22.57234 \nL 30.103125 22.729286 \nL 34.40611 42.982035 \nL 38.709095 50.468358 \nL 43.01208 55.311448 \nL 47.315065 58.54796 \nL 51.61805 61.373728 \nL 51.803125 61.503691 \nL 56.10611 76.016394 \nL 60.409095 76.736527 \nL 64.71208 74.503922 \nL 69.015065 80.722087 \nL 73.31805 85.393394 \nL 73.503125 85.545955 \nL 77.80611 106.106432 \nL 82.109095 106.818255 \nL 86.41208 107.538118 \nL 90.715065 108.310657 \nL 95.01805 108.95891 \nL 95.203125 108.981143 \nL 99.50611 112.424186 \nL 103.809095 113.043049 \nL 108.11208 113.775871 \nL 112.415065 114.147883 \nL 116.71805 114.350126 \nL 116.903125 114.378648 \nL 121.20611 116.823485 \nL 125.509095 117.16773 \nL 129.81208 117.174639 \nL 134.115065 117.534677 \nL 138.41805 117.860599 \nL 138.603125 117.85052 \nL 142.90611 118.926426 \nL 147.209095 119.229202 \nL 151.51208 119.870531 \nL 155.815065 119.928553 \nL 160.11805 120.026827 \nL 160.303125 120.038251 \nL 164.60611 120.951501 \nL 168.909095 121.003421 \nL 173.21208 121.355557 \nL 177.515065 121.424431 \nL 181.81805 121.773085 \nL 182.003125 121.753406 \nL 186.30611 122.514795 \nL 190.609095 122.848171 \nL 194.91208 122.760261 \nL 199.215065 122.815397 \nL 203.51805 122.970583 \nL 203.703125 122.971845 \nL 208.00611 123.49518 \nL 212.309095 123.916257 \nL 216.61208 124.046843 \nL 220.915065 124.110225 \nL 225.21805 124.116863 \nL 225.403125 124.110033 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path clip-path=\"url(#p18a32d5f07)\" d=\"M 12.70611 136.922727 \nL 17.009095 134.8001 \nL 21.31208 134.078263 \nL 25.615065 131.938959 \nL 29.91805 130.48671 \nL 30.103125 130.415379 \nL 34.40611 123.853439 \nL 38.709095 121.185265 \nL 43.01208 119.195252 \nL 47.315065 117.689242 \nL 51.61805 116.358729 \nL 51.803125 116.297384 \nL 56.10611 109.855055 \nL 60.409095 109.61206 \nL 64.71208 110.898501 \nL 69.015065 109.271391 \nL 73.31805 107.96351 \nL 73.503125 107.922062 \nL 77.80611 102.074468 \nL 82.109095 101.712359 \nL 86.41208 101.334368 \nL 90.715065 100.960744 \nL 95.01805 100.648901 \nL 95.203125 100.639502 \nL 99.50611 98.805955 \nL 103.809095 98.724957 \nL 108.11208 98.356495 \nL 112.415065 98.304481 \nL 116.71805 98.239921 \nL 116.903125 98.226161 \nL 121.20611 97.27652 \nL 125.509095 97.197904 \nL 129.81208 97.119288 \nL 134.115065 97.031143 \nL 138.41805 96.863906 \nL 138.603125 96.866825 \nL 142.90611 96.423657 \nL 147.209095 96.304542 \nL 151.51208 96.145722 \nL 155.815065 96.137781 \nL 160.11805 96.066312 \nL 160.303125 96.056707 \nL 164.60611 95.961491 \nL 168.909095 95.789966 \nL 173.21208 95.608911 \nL 177.515065 95.580323 \nL 181.81805 95.449773 \nL 182.003125 95.450772 \nL 186.30611 95.013336 \nL 190.609095 94.963308 \nL 194.91208 95.019689 \nL 199.215065 94.959735 \nL 203.51805 94.926621 \nL 203.703125 94.93275 \nL 208.00611 94.741754 \nL 212.309095 94.722696 \nL 216.61208 94.614698 \nL 220.915065 94.589287 \nL 225.21805 94.599769 \nL 225.403125 94.59717 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path clip-path=\"url(#p18a32d5f07)\" d=\"M 30.103125 116.786102 \nL 51.803125 112.384804 \nL 73.503125 101.653804 \nL 95.203125 99.016428 \nL 116.903125 96.719359 \nL 138.603125 96.878169 \nL 160.303125 95.18798 \nL 182.003125 97.053994 \nL 203.703125 94.382588 \nL 225.403125 94.949765 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 140.634375 59.234375 \nL 218.403125 59.234375 \nQ 220.403125 59.234375 220.403125 57.234375 \nL 220.403125 14.2 \nQ 220.403125 12.2 218.403125 12.2 \nL 140.634375 12.2 \nQ 138.634375 12.2 138.634375 14.2 \nL 138.634375 57.234375 \nQ 138.634375 59.234375 140.634375 59.234375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_22\">\n     <path d=\"M 142.634375 20.298438 \nL 162.634375 20.298438 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_23\"/>\n    <g id=\"text_11\">\n     <!-- train loss -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n      <path id=\"DejaVuSans-32\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n     </defs>\n     <g transform=\"translate(170.634375 23.798438)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"353.515625\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"405.615234\" xlink:href=\"#DejaVuSans-115\"/>\n     </g>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 142.634375 34.976562 \nL 162.634375 34.976562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_25\"/>\n    <g id=\"text_12\">\n     <!-- train acc -->\n     <g transform=\"translate(170.634375 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"325.830078\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"380.810547\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 142.634375 49.654688 \nL 162.634375 49.654688 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_27\"/>\n    <g id=\"text_13\">\n     <!-- test acc -->\n     <g transform=\"translate(170.634375 53.154688)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"100.732422\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"152.832031\" xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"192.041016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"223.828125\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"285.107422\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"340.087891\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p18a32d5f07\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"30.103125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```
