# Repro-Diffusion-Models

Reproduce Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [ ] DDIM



## Training

1. Edit the configuration file `./configs/config_ddpm.yml`.

2. Run command:

   ```shell
   # For single GPU/CPU
   python train.py --cfg ./configs/config_ddpm.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS train.py --cfg ./configs/config_ddpm.yml
   ```

   An experiment directory will be created under `./runs/` for each run.



## Sampling

1. Edit the configuration file `./configs/config_ddpm.yml`. 

2. To sample random images, run command:

   ```shell
   # For single GPU/CPU
   python test.py sample --cfg ./configs/config_ddpm.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS test.py sample --cfg ./configs/config_ddpm.yml
   ```

3. To sample images with denoising process, run command:

   ```shell
   # For single GPU/CPU
   python test.py sample_denoise --cfg ./configs/config_ddpm.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS test.py sample_denoise --cfg ./configs/config_ddpm.yml
   ```



## Evaluation

1. Sample random images (around 10k~50k images)

2. Edit the configuration file `./configs/config_ddpm.yml`. 

3. Run command:

   ```shell
   # For single GPU/CPU
   python test.py evaluate --cfg ./configs/config_ddpm.yml
   # For multiple GPUs
   torchrun --nproc_per_node NUM_GPUS test.py evaluate --cfg ./configs/config_ddpm.yml
   ```



## Results



### DDPM

**Quantitative results:**

|     Dataset     |  FID   |       IS        |
| :-------------: | :----: | :-------------: |
| CIFAR10 (32x32) | 3.1246 | 9.3690 (0.1015) |

**Qualitative results**:

<table width=100%>
  <tr>
    <th width=10% align="center">Dataset</th>
    <th width=40% align="center">MNIST</th>
    <th width=40% align="center">CIFAR-10</th>
  </tr>
  <tr>
    <th align="center">Random samples</th>
    <td align="center"><img src="./assets/ddpm-mnist-random.png"/></td>
    <td align="center"><img src="./assets/ddpm-cifar10-random.png"/></td>
  </tr>
  <tr>
    <th align="center">Denoising process</th>
    <td align="center"><img src="./assets/ddpm-mnist-denoise.png"/></td>
    <td align="center"><img src="./assets/ddpm-cifar10-denoise.png"/></td>
  </tr>
 </table>

 
