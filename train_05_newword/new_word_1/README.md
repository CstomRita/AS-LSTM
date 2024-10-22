# 此部分是新词识别功能

## 新词识别
采用基于互信息和左右熵的方式进行无监督学习

1. 根据N-Gram对词进行划分, 其中N的取值为1、2、3和4并统计共现词频。词频超过一定阈值，认为可能是一个词。
2. 根据凝固度公式计算2grams以上词语的凝固度，并过滤小于凝固度阈值的词语，针对不同长度的词语，凝固度的阈值设置不一样。进行初步过滤
3. 根据2得到的结果对句子粗略分词



## 选择标准

### 互信息

第一是最小互信息，因为互信息越大说明相关度越大，将n-gram分好的词计算互信息，如果低于阈值，则说明不能成词。

### 左右熵

第二是最小熵值，因为熵也是越大说明周边词越丰富，计算其左熵和右熵的最小值，如果最小值低于阈值，则说明不能成词。

### 最少出现次数

第三个是最少出现次数，为什么有这个数呢？假设前后两个词是完全相关的，出现400次，总共8000词，那么互信息=log((400/8000)/(400/8000)*(400/8000))，约掉之后剩下log(8000/400)。但是一个词如果从头到尾出现了一次，但是并不是单词，则互信息为=log((1/8000)/(1/8000)*(1/8000))=log(8000/1)，那么它的互信息会更大。



<font color=red>**取最少出现次数也会出现问题，就是一些低频率的词不能发现。**</font>

