#!/bin/sh

for i in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$i python train.py -i $i &
done
