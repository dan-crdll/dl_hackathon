#!/bin/bash

for split in A B C D; do
    python main.py --train-path data/$split/train.json.gz --test-path data/$split/test.json.gz --epochs 100
done

python zipthefolder.py
