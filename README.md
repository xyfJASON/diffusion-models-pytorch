# Repro-Diffusion-Models

Reproduce Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [ ] DDIM



## Training

Edit the configuration file `./configs/config_ddpm.yml`. 

For single GPU / CPU, run command:

```shell
python train.py --cfg ./configs/config_ddpm.yml
```

 For multiple GPUs, run command:

```shell
torchrun --nproc_per_node NUM_GPUS train.py --cfg ./configs/config_ddpm.yml
```

An experiment directory will be created under `./runs/` for each run.



## Evaluation

Edit the configuration file `./configs/config_ddpm_test.yml`. 

For single GPU / CPU, run command:

```shell
python test.py evaluate --cfg ./configs/config_ddpm_test.yml
```

 For multiple GPUs, run command:

```shell
torchrun --nproc_per_node NUM_GPUS test.py evaluate --cfg ./configs/config_ddpm_test.yml
```



## Sampling

Edit the configuration file `./configs/config_ddpm_test.yml`. 

To sample random images, run command:

```shell
python test.py sample --cfg ./configs/config_ddpm_test.yml
```

To sample images with denoising process, run command:

```shell
python test.py sample_denoise --cfg ./configs/config_ddpm_test.yml
```



## Results



### DDPM

:warning: Didn't get expected FID and IS scores as reported in paper.

|     Dataset     |   FID   |       IS        |
| :-------------: | :-----: | :-------------: |
| CIFAR10 (32x32) | 13.8240 | 8.5805 (0.1623) |



<img src="./assets/ddpm-mnist-random.png" width=25% /> <img src="./assets/ddpm-mnist-denoise.png" width=65% />

<img src="./assets/ddpm-cifar10-random.png" width=25% /> <img src="./assets/ddpm-cifar10-denoise.png" width=65% />
