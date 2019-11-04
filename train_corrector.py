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

from sklearn.decomposition import KernelPCA, PCA

from dataset.kernel_image_pair import KernelImagePair, default_augmentations, default_transforms
from network.sftmd import SFTMD, Predictor, Corrector
from common import tensor2img, get_datasets
from radam import RAdam

args = PinkBlack.io.setup(trace=False, default_args=dict(
    sftmd="ckpt/sftmd/sftmd.pth",
    nf=64,
    predictor="ckpt/predictor/predictor.pth",
    ckpt="ckpt/corrector/corrector.pth",
    gpu="0",
    batch_size=24,
    train="/",
    test="/",
    train_kernel="/",
    test_kernel="/",
    num_step=10000,
    validation_interval=500,
    num_workers=4,
    lr=0.0001,
    lr_decay=0.5,
    lr_min=1e-7,
    lr_scheduler="no", # 또는 'plateau' 또는 'no'
    optimizer='adam',
    loss="l2",
    metric="psnr",
    resume=False,
    seed=940513,
    scale=2,
    use_flickr=False,
    use_set5=False,
    use_urban100=False,
    patch_size=144,
    use_noise=False,
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

# valid_dl, set5_dl = set5_dl, valid_dl # To cherry-pick
# valid_dl, set5_dl = set5_dl, valid_dl # To cherry-pick

# ------------------------------------------------------
# Define loss and metrics

# loss
if args.loss == "l1":
    loss = F.l1_loss
elif args.loss == "l2":
    loss = F.mse_loss
else:
    raise NotImplementedError    

def criterion(output, bd):
    k_estimated = output
    k_gt = bd['k_reduced_gt']
    return loss(k_estimated, k_gt)

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


sftmd = SFTMD(nf=args.nf, scale=args.scale).cuda()
sftmd.load_state_dict(torch.load(args.sftmd))
sftmd.eval()
predictor = Predictor(datasets['train_dataset'].pca).cuda()
predictor.load_state_dict(torch.load(args.predictor))
predictor.eval() 

corrector = Corrector(nf=args.nf).cuda()

if args.optimizer == "radam":
    optimizer = RAdam(filter(lambda x: x.requires_grad, corrector.parameters()), lr=args.lr)
else:
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, corrector.parameters()), lr=args.lr)

if args.lr_scheduler == "multi":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [(x+1) * 30000 // args.validation_interval for x in range(10)], gamma=args.lr_decay)
else:
    scheduler = None
# trainer template code가 없어서 그냥 직접 짜야겠다..

trainer = Trainer(
    corrector,
    criterion=criterion,
    metric=metric,
    train_dataloader=train_dl,
    val_dataloader=valid_dl,
    test_dataloader=test_dl,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    ckpt=args.ckpt, 
    is_data_dict=True,
    logdir=args.ckpt+"_tb/",
    experiment_name=os.path.splitext(os.path.basename(args.ckpt))[0])

if args.use_set5: 
    trainer.dataloader['set5'] = set5_dl
                                    
if args.use_urban100:
    trainer.dataloader['urban100'] = urban100_dl

def _step_7(self, phase, iterator, only_inference=False):
    batch_dict = next(iterator)
    batch_size = batch_dict[list(batch_dict.keys())[0]].size(0)
    for k, v in batch_dict.items():
        batch_dict[k] = v.to(self.device)

    batch_dict['k_reduced_gt'] = batch_dict['k_reduced']
    with torch.no_grad():
        batch_dict['k_reduced'] = predictor(batch_dict, recon_kernel=False)
    
    for i in range(7):
        with torch.no_grad():
            batch_dict['SR'] = sftmd(batch_dict)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
            outputs = self.net(batch_dict)
            if not only_inference:
                loss = self.criterion(outputs, batch_dict)

            if only_inference:
                return outputs

            if phase == "train":
                loss.backward(retain_graph=True)
                self.optimizer.step()

        with torch.no_grad():
            batch_dict['k_reduced'] = outputs

    with torch.no_grad():
        metric = self.metric(batch_dict['SR'], batch_dict)

    return {'loss': loss.item(),
            'batch_size': batch_size,
            'metric': metric.item()}

Trainer._step = _step_7

os.makedirs(args.ckpt + "_result_imgs/", exist_ok=True)
def validation_callback():
    # To save result image - LR(nearest), SR, HR(GT)
    with torch.no_grad(): 
        for image_idx, bd in enumerate(set5_dl):        
            for k,v in bd.items():
                bd[k] = v.cuda()    
            
            lr = bd['LR'][0]
            w, h = bd['HR'].shape[2:]
        
            bd['k_reduced'] = predictor(bd, recon_kernel=False)
            srs = []
            metrics = []
            for i in range(7):
                sr = sftmd(bd)
                bd['SR'] = sr
                bd['k_reduced'] = corrector(bd)
                srs.append(sr[0])
                metrics.append(metric(sr, bd).item())
            srs.append(bd['HR'][0])
            
            lr = tensor2img(lr)
            lr = cv2.resize(lr, (h, w), interpolation=cv2.INTER_NEAREST)
            cat = torch.cat(srs, dim=1)
            cat = tensor2img(cat)
            img = np.concatenate((lr, cat), axis=0)
            
            Image.fromarray(img).save(trainer.ckpt + f"_result_imgs/{trainer.config['step']:08d}_{image_idx}.png")
            
            os.makedirs(trainer.ckpt + f"_result_imgs/_{trainer.config['step']:08d}/", exist_ok=True)
            Image.fromarray(lr).save(trainer.ckpt + f"_result_imgs/_{trainer.config['step']:08d}/{image_idx}_0_lr_.png")
            for num_step, sr in enumerate(srs[:-1]):
                img = tensor2img(sr)
                Image.fromarray(img).save(trainer.ckpt + f"_result_imgs/_{trainer.config['step']:08d}/{image_idx}_{num_step + 1}_sr_{image_idx}_{metrics[num_step]:.4f}.png")
            hr = srs[-1]
            img = tensor2img(hr)
            Image.fromarray(img).save(trainer.ckpt + f"_result_imgs/_{trainer.config['step']:08d}/{image_idx}_{9}_hr.png")


trainer.register_callback(validation_callback)

trainer.train(step=args.num_step, validation_interval=args.validation_interval, save_every_validation=True)
