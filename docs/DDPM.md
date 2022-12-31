# DDPM

> Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.



## Overview

The code supports training, sampling and evaluating, which share a common running command:

```shell
python main.py ddpm FUNC -c CONFIG
```

Or change `python` to `torchrun --nproc_per_node NUM_GPUS` if running on multiple GPUs.

The arguments are:

- `FUNC`: specify a function in the following choices:
  - `train`: to train DDPM
  - `sample`: to sample images
  - `sample_denoise`: to sample images with their denoising processes
  - `sample_skip`: to sample images with fewer steps
  - `evaluate`: to evaluate FID and IS
- `CONFIG`: path to the configuration file. All the configuration (data directory, hyper-parameters, etc.) are set in this file.

Below are specific instructions for training, sampling and evaluating.



## Training

1. Edit the configuration file `./configs/ddpm_cifar10.yml`.

2. Run command:

   ```shell
   # For single GPU/CPU
   python main.py ddpm train -c ./configs/ddpm_cifar10.yml
   ```
   
   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddpm train -c ./configs/ddpm_cifar10.yml
   ```
   
   An experiment directory will be created under `./runs/` for each run.



## Sampling

1. Edit the configuration file `./configs/ddpm_cifar10.yml`. 

2. To sample random images, run command:

   ```shell
   # For single GPU/CPU
   python main.py ddpm sample -c ./configs/ddpm_cifar10.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddpm sample -c ./configs/ddpm_cifar10.yml
   ```

3. To sample random images with fewer timesteps, run command:

   ```shell
   # For single GPU/CPU
   python main.py ddpm sample_skip -c ./configs/ddpm_cifar10.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddpm sample_skip -c ./configs/ddpm_cifar10.yml
   ```

4. To sample images with denoising process, run command:

   ```shell
   # For single GPU/CPU
   python main.py ddpm sample_denoise -c ./configs/config_ddpm.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddpm sample_denoise -c ./configs/config_ddpm.yml
   ```



## Evaluation

1. Sample random images (around 10k~50k images)

2. Edit the configuration file `./configs/ddpm_cifar10.yml`. 

3. Run command:

   ```shell
   # For single GPU/CPU
   python main.py ddpm evaluate -c ./configs/ddpm_cifar10.yml
   ```
   
   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddpm evaluate -c ./configs/ddpm_cifar10.yml
   ```

