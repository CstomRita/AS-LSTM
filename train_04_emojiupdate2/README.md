### 优化
 使用的双向LSTM

 #### 结果
 
 nohup.out| all_out_lstm_out[-1] + 选择loss最小的模型 | 66.35|

-|-|-

newnohup.out|all_out_lstm_out[-1]+ 选择 valid_acc最大的模型|67.19| new_model.pt |

newnohup1.out| torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1) + 选择 valid_acc最大的模型| new_model1.pt|


newnohup_lastone.out|torch.cat([all_out_lstm_out[0], all_out_lstm_out[-1]], dim=1) + 最后训练的模型| new_model_last_one.pt