# AS-LSTM
## 项目介绍

毕设题目：

基于表情符号和新词识别的情感分析

[metas](https://github.com/CstomRita/metas)是以此算法为底层设计的可视化系统

## pytorch 注意力机制

### **前提**

1. 输入向量一个长度为N的一维数组 [x1,x2,x3...xn]
2. 一个额外和任务有关的变量(查询向量) q

### 分类

1. 软注意力机制

   ![image-20200304094326976](README.assets/image-20200304094326976.png)

   注意力分布αi为在给定任务相关的查询q时，第i个输入向量受注意的程度。

   公式是注意力机制中给定的：

   1. ![image-20200304094610656](README.assets/image-20200304094610656.png)
   2. ![image-20200304094625960](README.assets/image-20200304094625960.png)

### 实现

根据业务场景，选择软注意力机制，我们要探究的是在表情符作用下，某个词的重要程度。

1. input[x1,x2,x3...xn] 经过embed，并经过LSTM

```python
embedded = self.embedding(input).view(1, 1, -1)
lstm_out = self.lstm(embedded)
```

2. 获取输入embedding和查询向量q的相似度,让全连接层自己来学习,把输入和查询向量连接在一起经过线性层

   ```python
   self.attn = nn.Linear(self.hidden_size*2, 1)
   similarity=self.attn(torch.cat((embedded, q), 1))
   ```

```python
for i in range(n):
	
```



3. 经过softmax得到权重矩阵即α矩阵

   ```pyrhon
   attn_weights = F.softmax(similarity, dim=1)
   ```

4. 权重矩阵作用在输入经过lstm的输出向量上，获得attention结果  (bmm：*对应相乘再相加*)，得到上图显示的最终a向量

   ```python
   attn_applied = torch.bmm(attn_weights.unsqueeze(0),lstm_out.unsqueeze(0))
   ```

> 平时搜到的代码都是关于机器学习的
>
> ![image-20200304111103391](README.assets/image-20200304111103391.png)
>
> 但是这个代码是比我当前应用场景复杂的
>
> 它改变的是C，再用C 和 输入 cat做运算
>
> 当前我的业务场景和上图Attention机制是一样的，根据Q得到权重矩阵，并乘积相加即可

## 项目结构

### train_01

去除表情符后，利用传统lstm，获取文本语义--->分类

数据集格式：tensor.sentence_no_emoji：分词后的一维数组

### train_02_textemoji

文本 + 表情符号 拼接后一起加入LSTM中

数据集格式：

1. tensor.sentence_no_emoji：分词后的一维数组
2. tensor.emoji：各个分句的表情符号，二维数组----->在train.py中reshape成一维数组，再传入模型

模型训练：

模型入参：sentences：一维数组；all_emojis：一维数组

### train_03_emojiorigin

传统表情符注意力机制

 











数据集格式：

1. tensor.sentence_no_emoji：分词后的一维数组
2. tensor.emoji：各个分句的表情符号，二维数组----->在train.py中reshape成一维数组，再传入模型

模型训练：

模型入参：sentences：一维数组；all_emojis：一维数组

流程：

1. alll_emojis的词向量获取平均值-----> emoji_ave_embedding  (1x1x300)
2. sentences 和 emoji_ave_embedding 做带注意力机制的LSTM

### train_03_emojiorigin_updateAttention

传统表情符注意力机制

数据集格式：

1. tensor.sentence_no_emoji：分词后的一维数组
2. tensor.emoji：各个分句的表情符号，二维数组----->在train.py中reshape成一维数组，再传入模型

模型训练：

模型入参：sentences：一维数组；all_emojis：一维数组

流程：

1. alll_emojis的词向量获取平均值-----> emoji_ave_embedding  (1x1x300)
2. sentences 和 emoji_ave_embedding 做带注意力机制的LSTM
   1. 利用sentences的词向量和emoji_ave_embedding的相似度得到权重矩阵
   2. sentences经过LSTM的输出和权重矩阵相乘

## 运行

```shell

CUDA_VISIBLE_DEVICES=0 python -u train.py 4 0 1

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 5 0 1 > result_crf_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 0 1> result_有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 0 1> result_多个分句有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 0 1> result_所有语料_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 0 1> result_有分句_最后一次.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 1 1 > result_有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 1 1> result_多个分句有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 1 1> result_所有语料_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 1 1> result_有分句_最优acc.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 2 1 > result_有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 2 1> result_多个分句有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 2 1> result_所有语料_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 2 1> result_有分句_最优loss.out 2>&1 &
```



查看

nvidia-smi

gpustat