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
