#!/bin/bash

SET=$(seq 0 499)
for stamp in $SET
do
	python main.py \
	--model_name AnyNetXA \
	--stamp $stamp \
	--dataset imagenet \
	--batch_size 128 \
	--epochs 10 \
	--optimizer sgd \
	--lr 0.05 \
	--checkpoint \
	--history \
	--data_path /workspace/data/Dataset/imagenet \
	--gpus 0,1
done
