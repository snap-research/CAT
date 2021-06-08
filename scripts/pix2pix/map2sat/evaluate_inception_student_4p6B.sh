#!/usr/bin/env bash
python profile.py --dataroot database/maps \
  --distiller inception \
  --log_dir evaluation_logs/pix2pix/map2sat/inception/student/4p6B \
  --restore_teacher_G_path logs/pix2pix/map2sat/inception/teacher/checkpoints/best_net_G.pth \
  --pretrained_student_G_path logs/pix2pix/map2sat/inception/student/4p6B/checkpoints/best_net_G.pth \
  --real_stat_path real_stat/maps_A.npz \
  --teacher_netG inception_9blocks --student_netG inception_9blocks \
  --pretrained_ngf 64 --teacher_ngf 64 --student_ngf 32 \
  --norm batch \
  --norm_affine \
  --norm_affine_D \
  --norm_track_running_stats \
  --channels_reduction_factor 6 \
  --kernel_sizes 1 3 5 \
  --direction BtoA \
  --batch_size 8 \
  --eval_batch_size 2 \
  --gpu_ids 0 \
  --num_threads 8 \
  --prune_cin_lb 16 \
  --target_flops 4.6e9 \

python metric/kid_score.py \
  --real evaluation_logs/pix2pix/map2sat/inception/student/4p6B/eval/0/real/ \
  --fake evaluation_logs/pix2pix/map2sat/inception/student/4p6B/eval/0/Sfake/ \
  --gpu 0
