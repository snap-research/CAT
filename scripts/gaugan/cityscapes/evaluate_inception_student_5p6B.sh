#!/usr/bin/env bash
python profile.py --dataroot database/cityscapes-origin \
  --distiller spade \
  --log_dir evaluation_logs/gaugan/cityscapes/inception/student/5p6B \
  --restore_teacher_G_path logs/gaugan/cityscapes/inception/teacher/checkpoints/best_net_G.pth \
  --pretrained_student_G_path logs/gaugan/cityscapes/inception/student/5p6B/checkpoints/best_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/table.txt \
  --teacher_netG inception_spade --student_netG inception_spade \
  --pretrained_netG inception_spade \
  --pretrained_ngf 64 --teacher_ngf 64 --student_ngf 48 \
  --teacher_norm_G spadesyncbatch3x3 --student_norm_G spadesyncbatch3x3 \
  --pretrained_norm_G spadesyncbatch3x3 \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --batch_size 8 \
  --gpu_ids 0 \
  --num_threads 8 \
  --target_flops 5.6e9 \
  --prune_cin_lb 16 \

python metric/kid_score.py \
  --real evaluation_logs/gaugan/cityscapes/inception/student/5p6B/eval/0/real/ \
  --fake evaluation_logs/gaugan/cityscapes/inception/student/5p6B/eval/0/Sfake/ \
  --gpu 0
