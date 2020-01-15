# upload至GPU服务器脚本
#! /bin/bash/expect
set timeout -1

spawn scp -r /Users/changsiteng/PycharmProjects/AS-LSTM-Version1 cst@computer12:~/cst
expect eof


spawn ssh computer12
expect eof

send "cd ~/cst/AS-LSTM-Version1/train_01\r"
expect eof
send "python3 train2.py\r"
expect eof
interact