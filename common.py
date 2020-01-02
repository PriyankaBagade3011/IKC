import numpy as np
import math
from torchvision.utils import make_grid
import os, glob
from sklearn.model_selection import train_test_split
from dataset.kernel_image_pair import KernelImagePair, default_augmentations, default_transforms

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, padding=0, nrow=1, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def get_datasets(args):
    train_imgs = glob.glob(args.train + "/**/*.png", recursive=True)
    test_imgs = glob.glob(args.test + "/**/*.png", recursive=True)
    train_imgs, valid_imgs = train_test_split(train_imgs, test_size=args.valid_rate, random_state=args.seed)

    if args.augment == "default":
        augmentations = default_augmentations
    elif args.augment == "custom":
        import imgaug as ia
        import imgaug.augmenters as iaa
        augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5), 
            iaa.Rot90([0, 1, 2, 3]),
            iaa.Sometimes(0.3, iaa.JpegCompression([0, 95])),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.1*255))),
        ])

    if args.use_flickr:
        flikr2k = glob.glob("../data/Flickr2K/Flickr2K_HR/*.png")
        train_imgs.extend(flikr2k)

    print(f"num of train {len(train_imgs)},  num of valid {len(valid_imgs)}.")
    print(f"num of test {len(test_imgs)}.")

    train_dataset = KernelImagePair(imgs=train_imgs, 
                                    kernel_pickle=args.train_kernel, scale=args.scale, 
                                    augmentations=default_augmentations, transforms=default_transforms, 
                                    patch_size=(args.patch_size, args.patch_size),
                                    seed=args.seed, train=True, noise=args.use_noise, interpolation=args.inter)

    valid_dataset = KernelImagePair(imgs=valid_imgs, 
                                    kernel_pickle=args.test_kernel, scale=args.scale, 
                                    augmentations=default_augmentations, transforms=default_transforms, 
                                    patch_size=(args.patch_size, args.patch_size),
                                    seed=args.seed, train=True, interpolation=args.inter)

    test_dataset = KernelImagePair(imgs=test_imgs, 
                                kernel_pickle=args.test_kernel, scale=args.scale, 
                                augmentations=default_augmentations, transforms=default_transforms, 
                                seed=args.seed, train=False, interpolation=args.inter)
    if args.use_set5:
        set5 = glob.glob("../data/testing_datasets/Set5/*.png")
        set5_dataset = KernelImagePair(imgs=set5,
                                    kernel_pickle=args.test_kernel, scale=args.scale, 
                                    augmentations=default_augmentations, transforms=default_transforms, 
                                    seed=args.seed, train=False, interpolation=args.inter)
        print(f"Set5 testset is available (num {len(set5)})")
    else:
        set5_dataset = None

    if args.use_urban100:
        urban100 = glob.glob("../data/testing_datasets/Urban100/*.png")
        urban100_dataset = KernelImagePair(imgs=urban100,
                                    kernel_pickle=args.test_kernel, scale=args.scale, 
                                    augmentations=default_augmentations, transforms=default_transforms, 
                                    seed=args.seed, train=False, interpolation=args.inter)
        print(f"Urban100 testset is available (num {len(urban100_dataset)})")
    else:
        urban100_dataset = None

    rt_dict = dict(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        set5_dataset=set5_dataset,
        urban100_dataset=urban100_dataset
    )
    
    return rt_dict