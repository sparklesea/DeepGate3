#!/bin/bash
NUM_PROC=2
GPUS=0,1

nohup python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./src/train_dg3.py \
 --exp_id plain \
 --gpus ${GPUS} \
 --data_dir ./data/dg3_all \
 --circuit_path ./data/dg3_all/graphs.npz \
 --pretrained_model_path ./trained/model_last.pth \
 --tf_arch plain \
 --batch_size 16 --fast \
 >> /uac/gds/zyzheng23/projects/DeepGate3-Transformer/exp/train_gate_path_graph.log 2>&1 &