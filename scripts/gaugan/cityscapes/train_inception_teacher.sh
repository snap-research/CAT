#!/usr/bin/env bash
python train.py --dataroot database/cityscapes-origin \
  --model spade --dataset_mode cityscapes \
  --log_dir logs/gaugan/cityscapes/inception/teacher \
  --netG inception_spade \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --gpu_ids 0,1,2,3,4,5,6,7 --load_in_memory --no_fid \
  --norm_G spadesyncbatch3x3 \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
