# DDIM

> Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." In *International Conference on Learning Representations*. 2020.



:information_source: Note: For distributed training / sampling / evaluating, replace `python` with `torchrun --nproc_per_node NUM_GPUS` in all the following commands.



## Training

DDIM shares the same training process with DDPM. Please refer to [DDPM doc](./DDPM.md).



## Sampling

1. Edit the configuration file (e.g. `./configs/ddim_cifar10.yml`).

2. To sample random images, run command:

   ```shell
   python main.py ddim sample -c CONFIG_FILE
   ```
   
3. To interpolate between two samples, run command:

   ```shell
   python main.py ddim sample_interpolate -c CONFIG_FILE
   ```
   



## Evaluation

1. Sample 10k~50k images.

2. Run command:

   ```shell
   python evaluate.py --dataset DATASET \
                      --dataroot DATAROOT \
                      --img_size IMG_SIZE \
                      --n_eval N_EVAL \
                      --fake_dir FAKE_DIR
   ```
   

