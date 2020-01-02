# TODO
# 2. sftmd 구조 변경 후 학습 
# 2 - 1. additional term 제거
# 2 - 2. sft-residual 개수 줄이기

# 1. degrad 고정 후 학습
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
import cv2
import matplotlib.pyplot as plt

from network.sftmd import SFTMD, Predictor, Corrector
from common import tensor2img, get_datasets
from radam import RAdam

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
    num_workers=4,
    lr=0.01,
    lr_decay=0.5,
    lr_min=1e-7,
    lr_scheduler="cosine", # 또는 'plateau' 또는 'no'
    optimizer='adam',
    loss="l2",
    metric="psnr",
    resume=False,
    seed=940513,
    scale=2,
    mode="SFTMD",
    use_flickr=False,
    use_set5=False,
    use_urban100=False,
    nf=64,
    patch_size=144,
    use_noise=False,
    valid_rate=0.1,
    inter="nearest",
    augment="default",
    kernel_dim=10,
    ))

PinkBlack.io.set_seeds(args.seed)

# --------------------------------------------------------- 
# Prepare training/validation/test data, and its dataloaders

datasets = get_datasets(args)
print(f"datasets are prepared.")

train_dl = DataLoader(datasets['train_dataset'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
valid_dl = DataLoader(datasets['valid_dataset'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
test_dl = DataLoader(datasets['test_dataset'], batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

if args.use_set5:
    set5_dl = DataLoader(datasets['set5_dataset'], batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

if args.use_urban100:
    urban100_dl = DataLoader(datasets['urban100_dataset'], batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

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
        # k_gt = bd['k'].view(bd['k'].shape[0], -1)
        k_gt = bd['k_reduced']
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
    net = SFTMD(input_para=args.kernel_dim, scale=args.scale, nf=args.nf).cuda()
elif args.mode == "PREDICTOR":
    net = Predictor(datasets['train_dataset'].pca, code_len=args.kernel_dim).cuda()
else:
    raise ValueError(f"network ?? {args.mode}")

if args.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr)
else:
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr)

if args.lr_scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=args.lr_min) # 2만번에 한 번 restart
elif args.lr_scheduler == "plateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=args.lr_decay)
elif args.lr_scheduler == "multi":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [(x+1) * 40000 // args.validation_interval for x in range(10)], gamma=args.lr_decay)
elif args.lr_scheduler == "no":
    scheduler = None
else:
    raise ValueError(f"unknown lr_scheduler :: {args.lr_scheduler}")

trainer = Trainer(net, 
                criterion=criterion, 
                metric=metric, 
                train_dataloader=train_dl, 
                val_dataloader=valid_dl, 
                test_dataloader=test_dl, 
                optimizer=optimizer, 
                lr_scheduler=scheduler,
                ckpt=args.ckpt, 
                is_data_dict=True,
                # clip_gradient_norm=3.,
                logdir=args.ckpt+"_tb/",
                experiment_name=os.path.splitext(os.path.basename(args.ckpt))[0]
                )

if args.use_set5: 
    trainer.dataloader['set5'] = set5_dl
                                    
if args.use_urban100:
    trainer.dataloader['urban100'] = urban100_dl
    
if args.mode == "SFTMD":
    os.makedirs(args.ckpt + "_result_imgs/", exist_ok=True)
    def validation_callback():
        # To save result image - LR(nearest), SR, HR(GT)
        with torch.no_grad(): 
            bd = next(iter(valid_dl))
            for k,v in bd.items():
                bd[k] = v.cuda()    
            
            sr = net(bd)

            hr = bd['HR']
            lr = bd['LR']

            cat = torch.cat([sr, hr], dim=3)
            img = tensor2img(cat.detach())
            
            lr_img = tensor2img(lr.detach())
            lr_img = cv2.resize(lr_img, (args.patch_size, args.batch_size * args.patch_size), interpolation=cv2.INTER_CUBIC)

            img = np.concatenate((lr_img, img), axis=1)
            
            Image.fromarray(img).save(trainer.ckpt + f"_result_imgs/{trainer.config['step']:08d}.png")
    # trainer.register_callback(validation_callback)
    
elif args.mode == "PREDICTOR":
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(args.ckpt + "_result_imgs/", exist_ok=True)

    mean_ = net.mean_
    components_ = net.components_
    def get_recon(flat):
        batch_mean = mean_.expand(flat.shape[0], 441)
        recon = torch.matmul(flat, components_) + batch_mean
        return recon
    
    def validation_callback():
        # To save result gt kernel - es kernel
        with torch.no_grad(): 
            bd = next(iter(valid_dl))
            for k,v in bd.items():
                bd[k] = v.cuda()    
            
            estimated = net(bd)
            estimated_recon = get_recon(estimated)
            gt_recon = get_recon(bd['k_reduced'])

            fig = plt.figure()
            plt.imshow(estimated_recon[0].view(21, 21).cpu().numpy(), cmap="gray")
            plt.savefig(trainer.ckpt + f"_result_imgs/{trainer.config['step']:08d}_es.png")
            # plt.savefig(trainer.ckpt + f"_result_imgs/_es.png")
            plt.close(fig)

            fig = plt.figure()
            plt.imshow(gt_recon[0].view(21, 21).cpu().numpy(), cmap="gray")
            plt.savefig(trainer.ckpt + f"_result_imgs/{trainer.config['step']:08d}_gt.png")
            # plt.savefig(trainer.ckpt + f"_result_imgs/_gt.png")
            plt.close(fig)

    # trainer.register_callback(validation_callback)
    
if args.resume:
    trainer.load(args.ckpt)

print("trainer is ready.")

trainer.train(step=args.num_step, validation_interval=args.validation_interval)

