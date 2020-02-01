# AS-LSTM
## 项目介绍

毕设题目：

基于表情符号和新词识别的情感分析

[metas](https://github.com/CstomRita/metas)是以此算法为底层设计的可视化系统

## 运行

```shell
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 5 0 1 > result_crf_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 0 1> result_有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 0 1> result_多个分句有表情符_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 0 1> result_所有语料_最后一次.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 0 1> result_有分句_最后一次.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 11 > result_有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 1 1> result_多个分句有表情符_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 1 1> result_所有语料_最优acc.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 1 1> result_有分句_最优acc.out 2>&1 &

---------------------------

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 4 21 > result_有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 3 2 1> result_多个分句有表情符_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 2 2 1> result_所有语料_最优loss.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u train.py 1 2 1> result_有分句_最优loss.out 2>&1 &
```



查看

nvidia-smi

gpustat