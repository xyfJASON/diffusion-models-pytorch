# DDPM

> Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.



:information_source: Note: For distributed training / sampling / evaluating, replace `python` with `torchrun --nproc_per_node NUM_GPUS` in all the following commands.



## Training

1. Edit the configuration file (e.g. `./configs/ddpm_cifar10.yml`).

2. Run command:

   ```shell
   python main.py ddpm train -c CONFIG_FILE -e EXP_NAME
   ```

   An experiment directory will be created under `./runs/` for each run, which will be named after `EXP_NAME` or current time if `EXP_NAME` is not provided.



## Sampling

1. Edit the configuration file (e.g. `./configs/ddpm_cifar10.yml`).

2. To sample random images, run command:

   ```shell
   python main.py ddpm sample -c CONFIG_FILE
   ```

3. To sample random images with fewer timesteps, run command:

   ```shell
   python main.py ddpm sample_skip -c CONFIG_FILE
   ```

4. To sample images with denoising process, run command:

   ```shell
   python main.py ddpm sample_denoise -c CONFIG_FILE
   ```

5. To sample images with progressive generation, i.e. visualizing predicted x_0 in each reverse step, run command:

   ```shell
   python main.py ddpm sample_progressive -c CONFIG_FILE
   ```



## Evaluation

1. Sample 10k~50k images.

3. Run command:

   ```shell
   python evaluate.py --dataset DATASET \
                      --dataroot DATAROOT \
                      --img_size IMG_SIZE \
                      --n_eval N_EVAL \
                      --fake_dir FAKE_DIR
   ```
   

