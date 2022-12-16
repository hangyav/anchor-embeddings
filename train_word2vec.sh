#!/bin/sh
set -e

LG=$1

DATA_DIR=./data/
BIN_DIR=./word2vec/bin
TEXT_DATA=$DATA_DIR/${LG}_data_preprocessed.txt
VECTOR_DATA=models/${LG}wiki-vector.bin

if [ ! -e $VECTOR_DATA ]; then
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 1 -size 300 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 5
fi
