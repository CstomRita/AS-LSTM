#!/bin/bash
set -e

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

make

# 因为demo默认是下载网上的语料来训练的，因此如果要训练自己的语料，需要注释掉
#if [ ! -e text8 ]; then
#  if hash wget 2>/dev/null; then
#    wget http://mattmahoney.net/dc/text8.zip
#  else
#    curl -O http://mattmahoney.net/dc/text8.zip
#  fi
#  unzip text8.zip
#  rm text8.zip
#fi

# 将CORPUS设置成语料的名字
CORPUS=words_origin.txt
# 自定义一下词典的保存名称
VOCAB_FILE=words_vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
# 定义训练出vector的存储名称
SAVE_FILE=glove.words.300.vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
# 向量维度，修改为300
VECTOR_SIZE=300
MAX_ITER=15
# 窗口size 据说8为最佳
WINDOW_SIZE=8
BINARY=2
NUM_THREADS=8
X_MAX=10

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
if [ "$CORPUS" = 'text8' ]; then
   if [ "$1" = 'matlab' ]; then
       matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
   elif [ "$1" = 'octave' ]; then
       octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
   else
       echo "$ python eval/python/evaluate.py"
       python eval/python/evaluate.py
   fi
fi


# make && bash demo.sh 如无意外，最后可以得到一个vector.txt的文件即为训练好的结果