#!/bin/bash

for split in A B C D; do
    python dl_hackathon/main.py --train-path data/$split/train.json.gz --test-path data/$split/test.json.gz --epochs 10
done

python fl_hackathon/zipthefolder.py