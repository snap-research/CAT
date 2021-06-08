#!/usr/bin/env bash
python profile.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --distiller inception \
  --gan_mode lsgan \
  --log_dir evaluation_logs/cycle_gan/horse2zebra/inception/student/2p6B \
  --restore_teacher_G_path logs/cycle_gan/horse2zebra/inception/teacher/checkpoints/best_A_net_G_A.pth \
  --pretrained_student_G_path logs/cycle_gan/horse2zebra/inception/student/2p6B/checkpoints/best_net_G.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --teacher_netG inception_9blocks --student_netG inception_9blocks \
  --pretrained_ngf 64 --teacher_ngf 64 --student_ngf 20 \
  --ndf 64 \
  --norm_affine \
  --norm_affine_D \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --batch_size 8 \
  --eval_batch_size 2 \
  --gpu_ids 0 \
  --num_threads 8 \
  --prune_cin_lb 16 \
  --target_flops 2.6e9 \

python metric/kid_score.py \
  --real database/horse2zebra/testB \
  --fake evaluation_logs/cycle_gan/horse2zebra/inception/student/2p6B/eval/0/Sfake/ \
  --gpu 0
