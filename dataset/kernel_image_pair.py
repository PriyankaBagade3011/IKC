import imgaug as ia
import imgaug.augmenters as iaa
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from sklearn.decomposition import PCA

default_augmentations = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5), 
            iaa.Rot90([0, 1, 2, 3]),
            # iaa.Sometimes(0.3, iaa.JpegCompression([0, 95])),
            # iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.1*255))),
        ])

default_transforms = transforms.ToTensor()
        
KERNEL_SIZE = 21

class KernelImagePair(Dataset):
    def __init__(self, imgs:list, 
                 kernel_pickle:str,
                 scale:int,
                 augmentations:object,
                 transforms:object,
                 seed=0,
                 patch_size=(144, 144),
                 train=True, noise=False, cubic=False, interpolation="nearest", downsample_on_pipe=True):
        super(KernelImagePair, self).__init__()
        self._kernel_dict = torch.load(kernel_pickle)
        
        self.kernels = self._kernel_dict['kernels']  # N x (21*21) 2d
        self.kernel_size = (KERNEL_SIZE, KERNEL_SIZE)  # (21, 21)
        self.k_reduced = self._kernel_dict['k_reduced']  # N x 21  2d
        self.stddevs = self._kernel_dict['stddevs']  # N standard deviations

        self.pca = self._kernel_dict['pca']  # PCA object (sklearn.decomposition)
        
        self.imgs = imgs
        self.scale = scale
        self.augmentations = augmentations
        
        self.transforms = transforms
        self.seed = seed
        self.patch_size = patch_size
        self.train=train
        self.need_cubic = cubic
        self.downsample_on_pipe = downsample_on_pipe
        
        if interpolation == "nearest":
            self.inter = cv2.INTER_NEAREST
        elif interpolation == "cubic":
            self.inter = cv2.INTER_CUBIC

        self.random = np.random.RandomState(seed)
        self.noise = None
        if noise:
            self.noise = iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.1*255))
            
    def __getitem__(self, idx) -> dict:
        img = self.imgs[idx]
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        
        if self.train:
            """
            Random Crop image Adding margin w.r.t kernel_size and patch_size
            """
            img_from, img_to = np.zeros(2, dtype=int), np.zeros(2, dtype=int)
            for i in range(2):
                img_from[i] = self.random.randint(0, img.shape[i] - (self.kernel_size[i] + 1 + self.patch_size[i]))
                img_to[i] = img_from[i] + self.kernel_size[i] + 1 + self.patch_size[i]
            img_patch = img[img_from[0]:img_to[0], img_from[1]:img_to[1]]
        else:
            img_patch = img
        
        if self.train and self.augmentations is not None:
            img_patch = self.augmentations.augment_image(img_patch)
            img = img_patch
        
        # Select kernel randomly
        kernel_idx = self.random.randint(len(self.kernels))
        stddev = self.stddevs[kernel_idx]
        gaussian_kernel = self.kernels[kernel_idx].reshape(KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)
        if self.downsample_on_pipe:
            img_blur = cv2.filter2D(img_patch, ddepth=-1, kernel=gaussian_kernel)
        else:
            img_blur = img_patch
        k_reduced = self.k_reduced[kernel_idx].astype(np.float32)

        if self.train:            
            # edge effect가 발생하는 영역 cut
            half = self.kernel_size[0] // 2 + 1, self.kernel_size[1] // 2 + 1
            img_blur = img_blur[half[0] : -half[0], half[1]:-half[1]]
            img = img[half[0] : -half[0], half[1]:-half[1]]
            img_lr = cv2.resize(img_blur, 
                                (self.patch_size[0]//self.scale, self.patch_size[1]//self.scale), self.inter)
            if self.noise is not None:
                img_lr = self.noise.augment_image(img_lr)
        elif self.downsample_on_pipe:
            img_lr = cv2.resize(img_blur, 
                    (img.shape[1]//self.scale, img.shape[0]//self.scale), self.inter)
        else:
            img_lr = img_blur
            img = cv2.imread(self.imgs[idx].replace("/lr/", "/hr/"))

        if self.transforms is not None:
            if self.train:
                # 이걸 꼭 써야하나? 안하면 ToTensor 할 때 안되긴 함
                if not img_lr.flags['C_CONTIGUOUS']:
                    img_lr = np.ascontiguousarray(img_lr)
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
            if self.need_cubic:
                # demo 상황일 때 cubic resize된 샘플도 추가
                w, h = img.shape[:2]
                img_cubic = cv2.resize(img_lr, (h, w), cv2.INTER_CUBIC)
                img_cubic = self.transforms(img_cubic)

            img_lr = self.transforms(img_lr)
            img = self.transforms(img)
        

        re_dict = dict(LR=img_lr,
                    HR=img,
                    k=gaussian_kernel,
                    k_reduced=k_reduced,
                    stddev=stddev,
                    )
        
        if self.need_cubic:
            re_dict['lr_cubic'] = img_cubic
        
        return re_dict

    def __len__(self):
        return len(self.imgs)
