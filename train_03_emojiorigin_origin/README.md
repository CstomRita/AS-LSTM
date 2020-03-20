 train_03_emojiorigin_updateAttention
 
 是在train_03_emojiorigin的基础上
 
 更改了 加了注意力机制的LSTM 那一部分
 
 按照自己的理解进行编码
 
 更改的内容：
 
 1. lstm_emoji_attention部分的attention_net
 
    1. sentence_embeddings 和 emoji_ave_embeddingx做注意力机制
        - 通过看sentence_embeddings 和 emoji_ave_embedding相似度得到权重矩阵，通过expand emoji_ave_embedding至 length * 300
        - 再和sentence_embeddings_permute cat成 length * 600
        - 通过线性层 length * 600 ---> length * 1的一维矩阵代表各个单词的权重

    2. 权重矩阵和lstm_out相乘相加得到 1 X 1 X hidden_size
        - attn_weights---> 7X1---->需要改成 1X1X7的格式
        - lstm_out--> 7 * 1 * 128 改为 1 * 7 * 128的格式
 

 2. 在每个epoch的时候init_hidden(好像这个hidden的初始化并不会影响)
 
    > 遇到问题：
    > 如果在train.py里传递hidden，模型有两个输出，将报错` Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.`
    
    