python -u ./src/train_dg3.py --exp_id 0 --data_dir ./data --npz_dir ./dataset/ --pretrained_model_path ../DeeperGate/deepgate/pretrained/model.pth --tf_arch sparse --lr 1e-4 --workload --gpus 0 --batch_size 64 --epoch 200 --test_npz_path ./dataset/1p.npz