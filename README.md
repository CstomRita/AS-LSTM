# AS-LSTM
基于表情符号和新词识别的情感分析

运行

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 5 0 > result_以表情符为分隔符有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 0 > result_有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 0 > result_多个分句有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 0 > result_所有语料_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 0 > result_有分句_最后一次.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 1 > result_有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 1 > result_多个分句有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 1 > result_所有语料_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 1 > result_有分句_最优acc.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 2 > result_有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 2 > result_多个分句有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 2 > result_所有语料_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 2 > result_有分句_最优loss.out 2>&1 &

查看

nvidia-smi

gpustat