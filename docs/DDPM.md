# DDPM

> Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.



## Training

1. Edit the configuration file `./configs/ddpm_cifar10.yml`.

2. Run command:

   ```shell
   # For single GPU/CPU
   python main.py train -c ./configs/ddpm_cifar10.yml
   ```
   
   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py train -c ./configs/ddpm_cifar10.yml
   ```
   
   An experiment directory will be created under `./runs/` for each run.



## Sampling

1. Edit the configuration file `./configs/ddpm_cifar10.yml`. 

2. To sample random images, run command:

   ```shell
   # For single GPU/CPU
   python main.py sample -c ./configs/ddpm_cifar10.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py sample -c ./configs/ddpm_cifar10.yml
   ```

3. To sample random images with fewer timesteps , run command:

   ```shell
   # For single GPU/CPU
   python main.py sample_skip -c ./configs/ddpm_cifar10.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py sample_skip -c ./configs/ddpm_cifar10.yml
   ```

4. To sample images with denoising process, run command:

   ```shell
   # For single GPU/CPU
   python main.py sample_denoise -c ./configs/config_ddpm.yml
   ```

   ```shell
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS main.py sample_denoise -c ./configs/config_ddpm.yml
   ```



## Evaluation

1. Sample random images (around 10k~50k images)

2. Edit the configuration file `./configs/ddpm_cifar10.yml`. 

3. Run command:

   ```shell
   # For single GPU/CPU
   python test.py evaluate -c ./configs/ddpm_cifar10.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS test.py evaluate -c ./configs/ddpm_cifar10.yml
   ```


