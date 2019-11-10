# AS-LSTM
基于表情符号和新词识别的情感分析

运行
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 > result_多个分句有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 > result_所有语料_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 > result_有分句_最后一次.out 2>&1 &
