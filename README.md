# Blind SR with Iterative Kernel Correction (CVPR2019)

> Original paper : [https://arxiv.org/abs/1904.03377](https://arxiv.org/abs/1904.03377)
> 
> Implementation of original author(s) : https://github.com/yuanjunchai/IKC
----

## Important ##
- There are no plans to update this repository.
- If you're looking for a good implementation of this paper, please see the original implementation.

### PSNR Comparison ###

| LR(x4)   | BICUBIC | EDSR | IKC(0) | IKC(7) | GroundTruth |
| -------------------- | ----- | ---- | ---- | ----- | ----- |
| ![lr](test_imgs/set5/aniso_diverse/lr/butterfly.png) | ![sr_bicubic](test_imgs/set5/aniso_diverse/sr_bicubic/butterfly.png) | ![sr_edsr](test_imgs/set5/aniso_diverse/sr_edsr/butterfly_x4_SR.png) | ![sr_0](test_imgs/set5/aniso_diverse/sr_0/butterfly.png)  | ![sr_6](test_imgs/set5/aniso_diverse/sr_6/butterfly.png)  | ![hr](test_imgs/set5/aniso_diverse/hr/butterfly.png) |
| ![lr](test_imgs/set5/aniso_diverse/lr/bird.png) | ![sr_bicubic](test_imgs/set5/aniso_diverse/sr_bicubic/bird.png) | ![sr_edsr](test_imgs/set5/aniso_diverse/sr_edsr/bird_x4_SR.png) | ![sr_0](test_imgs/set5/aniso_diverse/sr_0/bird.png)  | ![sr_6](test_imgs/set5/aniso_diverse/sr_6/bird.png)  | ![hr](test_imgs/set5/aniso_diverse/hr/bird.png) |
| - | 28.33 | 32.26 | - | 33.04 | - |


### Test on Real Image ###

| LR   | BICUBIC | EDSR | IKC
| --- | --- | --- | --- |
| ![lr](test_imgs/real/chip_clr_original.png) | ![cubic](test_imgs/real/chip_clr.png) | ![edsr](test_imgs/real/chip_edsr.png) | ![cubic](test_imgs/real/chip_csr_07.png) |
