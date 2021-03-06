python train.py --train ../data/DIV2K/DIV2K_train_HR/\
 --test ../data/DIV2K/DIV2K_valid_HR/ \
--train_kernel kernels/train/kernel_scale4_iso_dim10.pth \
--test_kernel kernels/test/kernel_scale4_iso_dim10.pth \
--lr 0.01 \
--ckpt ckpt/sftmd/191009_0_sftmd_x4.pth \
--loss l1 \
--gpu 1 \
--validation_interval 500 \
--num_step 200000 \
--metric psnr \
--lr_decay 0.4 \
--use_flickr \
--use_set5 \
--use_urban100 \
--scale 4
