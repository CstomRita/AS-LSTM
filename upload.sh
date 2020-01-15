# upload至GPU服务器脚本
#! /bin/bash/expect
# 10.112.24.79 cst csttsc
set timeout -1

spawn scp -r /Users/changsiteng/PycharmProjects/AS-LSTM-Version1 cst@gpu:~/sentimentAnalyze
expect eof


spawn ssh cst@10.112.24.79
send "cd ~/sentimentAnalyze/AS-LSTM-Version1/train_01\r"
send "python train2.py\r"
interact