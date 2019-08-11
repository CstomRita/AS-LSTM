
#! /bin/bash/expect

#使用Annocada构建本工程需要的环境
# annocada是一个完全隔离的环境，离开这个环境的Python版本、任何东西都可以不一样，他们互相隔离
# Python3.7 pytorch1.1.0

set timeout -1

# 安装Annocada
spawn wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
expect eof

spawn bash Anaconda3-2019.03-Linux-x86_64.sh
expect ">>>"
send "\r"
expect eof

send "\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r"
expect eof
send "yes\r"
expect eof
send "\r"
expect eof
send "yes\r"
expect eof

# pytorch 1.1.0
spawn pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
expect eof
spawn pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
expect eof

