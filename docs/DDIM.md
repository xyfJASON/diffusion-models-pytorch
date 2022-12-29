# DDIM

> Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." In *International Conference on Learning Representations*. 2020.



## Overview

The code supports training, sampling and evaluating, which share a common running command:

```shell
python main.py ddim FUNC -c CONFIG
```

Or change `python` to `torchrun --nproc_per_node NUM_GPUS` if running on multiple GPUs.

The arguments are:

- `FUNC`: specify a function in the following choices:
  - `sample`: to sample images
  - `evaluate`: to evaluate FID and IS
- `CONFIG`: path to the configuration file. All the configuration (data directory, hyper-parameters, etc.) are set in this file.

Below are specific instructions for training, sampling and evaluating.



## Training

DDIM shares the same training process with DDPM. Please refer to [DDPM doc](./DDPM.md).



## Sampling

1. Edit the configuration file `./configs/ddim_cifar10.yml`. 

2. To sample random images, run command:

   ```shell
   # For single GPU/CPU
   python main.py ddim sample -c ./configs/ddim_cifar10.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddim sample -c ./configs/ddim_cifar10.yml
   ```



## Evaluation

1. Sample random images (around 10k~50k images)

2. Edit the configuration file `./configs/ddim_cifar10.yml`. 

3. Run command:

   ```shell
   # For single GPU/CPU
   python main.py ddim evaluate -c ./configs/ddim_cifar10.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py ddim evaluate -c ./configs/ddim_cifar10.yml
   ```

