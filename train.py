# TODO
# 1. degrad 고정 후 학습
# 2. sftmd 구조 변경 후 학습 
# 2 - 1. additional term 제거
# 2 - 2. sft-residual 개수 줄이기
# 3. # of param 확인
# 4. predictor 출력 범위 제한

import PinkBlack.io
from PinkBlack.trainer import Trainer

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import os, glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

from dataset.kernel_image_pair import KernelImagePair, default_augmentations, default_transforms
from network.sftmd import SFTMD, Predictor, Corrector
from common import tensor2img

args = PinkBlack.io.setup(trace=False, default_args=dict(
    ckpt="ckpt/sftmd/sftmd.pth",
    gpu="0",
    batch_size=24,
    train="/",
    test="/",
    train_kernel="/",
    test_kernel="/",
    num_step=10000,
    validation_interval=500,
    num_workers=8,
    lr=0.01,
    lr_decay=0.5,
    loss="l1",
    metric="l1",  # TODO : ssim 등 추가하기
    resume=False,
    seed=940513,
    scale=3,
    mode="SFTMD",
    use_flickr=False,
    use_set5=False,
    use_urban100=False,
    ))

PinkBlack.io.set_seeds(args.seed)

# --------------------------------------------------------- 
# Prepare training/validation/test data

train_imgs = glob.glob(args.train + "/**/*.png", recursive=True)
test_imgs = glob.glob(args.test + "/**/*.png", recursive=True)

train_imgs, valid_imgs = train_test_split(train_imgs, test_size=0.2, random_state=args.seed)

# TODO 여러 degradation일때 지우기
args.test_kernel = args.train_kernel

if args.use_flickr:
    flikr2k = glob.glob("../data/Flickr2K/Flickr2K_HR/*.png")
    train_imgs.extend(flikr2k)

print(f"num of train {len(train_imgs)},  num of valid {len(valid_imgs)}.")
print(f"num of test {len(test_imgs)}.")

train_dataset = KernelImagePair(imgs=train_imgs, 
                                kernel_pickle=args.train_kernel, scale=args.scale, 
                                augmentations=default_augmentations, transforms=default_transforms, 
                                seed=args.seed, train=True)

valid_dataset = KernelImagePair(imgs=valid_imgs, 
                                kernel_pickle=args.train_kernel, scale=args.scale, 
                                augmentations=default_augmentations, transforms=default_transforms, 
                                seed=args.seed, train=True)
# TODO :: validation을 patch로 할지 full size image로 할지

test_dataset = KernelImagePair(imgs=test_imgs, 
                            kernel_pickle=args.test_kernel, scale=args.scale, 
                            augmentations=default_augmentations, transforms=default_transforms, 
                            seed=args.seed, train=False)
if args.use_set5:
    set5 = glob.glob("../data/testing_datasets/Set5/*.png")
    set5_dataset = KernelImagePair(imgs=set5,
                                kernel_pickle=args.test_kernel, scale=args.scale, 
                                augmentations=default_augmentations, transforms=default_transforms, 
                                seed=args.seed, train=False)
    set5_dl = DataLoader(set5_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    print(f"Set5 testset is available (num {len(set5)})")

    
if args.use_urban100:
    urban100 = glob.glob("../data/testing_datasets/Urban100/*.png")
    urban100_dataset = KernelImagePair(imgs=urban100,
                                kernel_pickle=args.test_kernel, scale=args.scale, 
                                augmentations=default_augmentations, transforms=default_transforms, 
                                seed=args.seed, train=False)
    urban100_dl = DataLoader(urban100_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    print(f"Urban100 testset is available (num {len(urban100_dataset)})")
    
print(f"datasets are prepared.")

train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

# ------------------------------------------------------
# Define loss and metrics

# loss
if args.loss == "l1":
    loss = F.l1_loss
elif args.loss == "l2":
    loss = F.mse_loss
else:
    raise NotImplementedError    

if args.mode == "SFTMD":
    def criterion(output, bd):
        sr = output
        hr = bd['HR']
        return loss(sr, hr)
elif args.mode == "PREDICTOR":
    def criterion(output, bd):
        k_estimated = output
        k_gt = bd['k'].view(bd['k'].shape[0], -1)
        # k_reduced = bd['k']
        return loss(k_estimated, k_gt)
else:
    raise NotImplementedError

# Metric
if args.metric == "l1" or args.metric == "l2":
    def metric(output, bd):
        with torch.no_grad():
            return -criterion(output, bd)
elif args.metric == "psnr":
    def metric(output, bd):
        with torch.no_grad():
            sr = output
            hr = bd['HR']
            diff = sr - hr
            shave = args.scale + 10
            valid = diff[..., shave:-shave, shave:-shave]
            mse = valid.pow(2).mean()
            return -10 * torch.log10(mse)

else:
    raise NotImplementedError

# ------------------------------------------------------------- 

if args.mode == "SFTMD":
    sftmd = SFTMD(input_para=10, scale=args.scale).cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, sftmd.parameters()), lr=args.lr)

    if 0 < args.lr_decay < 1:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=args.lr_decay)
    else:
        scheduler = None

    trainer = Trainer(sftmd, 
                    criterion=criterion, 
                    metric=metric, 
                    train_dataloader=train_dl, 
                    val_dataloader=valid_dl, 
                    test_dataloader=test_dl, 
                    optimizer=optimizer, 
                    lr_scheduler=scheduler,
                    ckpt=args.ckpt, 
                    is_data_dict=True,
                    clip_gradient_norm=3.
                    )

    os.makedirs(args.ckpt + "_result_imgs/", exist_ok=True)
    def validation_callback():
        # To save result image - LR(nearest), SR, HR(GT)
        with torch.no_grad(): 
            bd = next(iter(valid_dl))
            for k,v in bd.items():
                bd[k] = v.cuda()    
            
            sr = sftmd(bd)
            hr = bd['HR']
            lr = bd['LR']

            cat = torch.cat([sr, hr], dim=3)
            img = tensor2img(cat.detach())
            
            lr_img = tensor2img(lr.detach())
            lr_img = cv2.resize(lr_img, (144, args.batch_size * 144), interpolation=cv2.INTER_NEAREST)

            img = np.concatenate((lr_img, img), axis=1)
            
            Image.fromarray(img).save(trainer.ckpt + f"_result_imgs/{trainer.config['step']:08d}.png")

    trainer.register_callback(validation_callback)

    if args.use_set5:
        trainer.dataloader['set5'] = set5_dl
                                      
    if args.use_urban100:
        trainer.dataloader['urban100'] = urban100_dl

    print("trainer is ready.")

    trainer.train(step=args.num_step, validation_interval=args.validation_interval)

elif args.mode == "PREDICTOR":
    predictor = Predictor(train_dataset.pca).cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, predictor.parameters()), lr=args.lr)

    if 0 < args.lr_decay < 1:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=args.lr_decay)
    else:
        scheduler = None

    p_trainer = Trainer(predictor, 
                        criterion=criterion,
                        metric=metric, 
                        train_dataloader=train_dl, 
                        val_dataloader=valid_dl, 
                        test_dataloader=test_dl, 
                        optimizer=optimizer, 
                        lr_scheduler=scheduler,
                        ckpt=args.ckpt, 
                        is_data_dict=True,
                        clip_gradient_norm=3.
                        )
    if args.use_set5:
        p_trainer.dataloader['set5'] = set5_dl
    print("p_trainer is ready.")
    p_trainer.train(step=args.num_step, validation_interval=args.validation_interval)

else:
    raise NotImplementedError

