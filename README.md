# Repro-Diffusion-Models

Reproduce Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [ ] DDIM



## Training

For single GPU / CPU, run command:

```shell
python train.py
```

 For multiple GPUs (e.g. 2 GPUs), run command:

```shell
torchrun --nproc_per_node 2 train.py
```



## Generation

Run command:

```shell
python generate.py \
    --model_path MODEL_PATH \
    --mode {random,denoise} \
    --save_path SAVE_PATH \
    --dim DIM \
    --n_stages N_STAGES \
    --img_size IMG_SIZE \
    --total_steps TOTAL_STEPS \
    [--img_channels IMG_CHANNELS] \
    [--beta_schedule_mode BETA_SCHEDULE_MODE] \
    [--cpu]
```



## Results



### DDPM

<img src="./assets/ddpm-mnist-random.png" width=30% /> <img src="./assets/ddpm-mnist-denoise.png" width=60% /> 

