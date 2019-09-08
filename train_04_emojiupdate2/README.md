### 优化
 在update1上的优化：使用的双向LSTM

 #### 结果
 | out结果 | 说明 |模型|准确率|
| -------- | -------- |--------|-------|
| nohup.out|all_out_lstm_out[-1] + 选择loss最小的模型|model.pt|66.35 |
|newnohup.out|all_out_lstm_out[-1]+ 选择 valid_acc最大的模型|new_model.pt |67.19|
|newnohup1.out| torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1) + 选择 valid_acc最大的模型| new_model1.pt|67.19|
|newnohup_lastone.out|torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1) + 最后训练的模型| new_model_last_one.pt|80.74|