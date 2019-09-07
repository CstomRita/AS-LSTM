 train_04是将表情符作为注意力机制向量加入的

 注意力机制向量的学习：

 1. 未特殊记录表情符的数量，直接按照表情符的文本学习

 2. 通过一个LSTM获取上下文向量

 3. 使用的是基本的Attention

 ![Attention机制](README.assets/attention.jpg)

 

 #### 模型

  ![train_04模型——emoji_attention](README.assets/model.jpg)

### 结果

nohup.out| all_out_lstm_out[-1] + 选择loss最小的模型 | 66.61
-|-|-
newnohup.out|torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1) + 选择 valid_acc最大的模型|66.95
|
