 train_03_emojiorigin是按照之前论文的记录，
 以整个句子为粒度,
 将表情符作为注意力机制向量加入：


1. 数据切分为：所有的单词，所有的表情符
2. 表情符语义向量为：表情符词向量的均值
3. 按照表情符语义向量 + 分词结果做注意力


结果：

| nohup.out|选择loss最小的模型|model.pt|67.23 |
| -------- | -------- |--------|-------|
|newnohup.out|选择 valid_acc最大的模型|new_model.pt |67.23|
|newnohup_last_one.out|选择 valid_acc最大的模型|new_model_last_one.pt ||