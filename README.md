# Repro-Diffusion-Models

Reproduce Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [ ] DDIM



## Training

For single GPU / CPU, run command:

```shell
python train.py --config_path CONFIG_PATH
```

 For multiple GPUs (e.g. 2 GPUs), run command:

```shell
torchrun --nproc_per_node 2 train.py --config_path CONFIG_PATH
```



## Generation

Run command:

```shell
python generate.py \
    --model_path MODEL_PATH \
    --mode {random,denoise} \
    --save_path SAVE_PATH \
    [--cpu] \
    [--img_channels IMG_CHANNELS] \
    [--img_size IMG_SIZE] \
    [--dim DIM] \
    [--dim_mults DIM_MULTS [DIM_MULTS ...]] \
    [--total_steps TOTAL_STEPS] \
    [--beta_schedule_mode BETA_SCHEDULE_MODE]
```



## Results



### DDPM

<img src="./assets/ddpm-mnist-random.png" width=30% /> <img src="./assets/ddpm-mnist-denoise.png" width=60% /> 

<img src="./assets/ddpm-celebahq-random.png" width=30% /> <img src="./assets/ddpm-celebahq-denoise.png" width=60% /> 

