#!/bin/sh

for i in {0..9}
do
    python train_rnd.py -o "model_ckpts_rnd_w$1" -w $1 -i $i
done