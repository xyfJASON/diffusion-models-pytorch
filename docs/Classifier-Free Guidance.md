# Classifier-Free Guidance

> Ho, Jonathan, and Tim Salimans. "Classifier-Free Diffusion Guidance." In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. 2021.



:information_source: Note: For distributed training / sampling, replace `python` with `torchrun --nproc_per_node NUM_GPUS` in all the following commands.



## Training

1. Edit the configuration file (e.g. `./configs/classifier_free_cifar10.yml`).

2. Run command:

   ```shell
   python main.py classifier_free train -c CONFIG_FILE -e EXP_NAME
   ```

   An experiment directory will be created under `./runs/` for each run, which will be named after `EXP_NAME` or current time if `EXP_NAME` is not provided.



## Sampling

1. Edit the configuration file (e.g. `./configs/classifier_free_cifar10.yml`).

2. To sample random images, run command:

   ```shell
   python main.py classifier_free sample -c CONFIG_FILE
   ```

