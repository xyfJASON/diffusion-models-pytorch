# Classifier-Free Guidance

> Ho, Jonathan, and Tim Salimans. "Classifier-Free Diffusion Guidance." In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. 2021.



## Training

This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the script on different platforms.

```shell
accelerate-launch scripts/train_ddpm_cfg.py -c CONFIG [-e EXP_DIR] [--key value ...]
```

Arguments:

- `-c CONFIG`: path to the training configuration file.
- `-e EXP_DIR`: results (logs, checkpoints, tensorboard, etc.) will be saved to `EXP_DIR`. Default to `runs/exp-{current time}/`.
- `--key value`: modify configuration items in `CONFIG` via CLI.

For example, to train on CIFAR-10 with default settings:

```shell
accelerate-launch scripts/train_ddpm_cfg.py -c ./configs/ddpm_cfg_cifar10.yaml
```

To change the default `p_uncond` (the probability to disable condition in training) in `./configs/ddpm_cfg_cifar10.yaml` from 0.2 to 0.1:

```shell
accelerate-launch scripts/train_ddpm_cfg.py -c ./configs/ddpm_cfg_cifar10.yaml --train.p_uncond 0.1
```



## Sampling

```shell
accelerate-launch scripts/sample_cfg.py -c CONFIG \
                                        --weights WEIGHTS \
                                        --sampler {ddpm,ddim} \
                                        --n_samples_each_class N_SAMPLES_EACH_CLASS \
                                        --save_dir SAVE_DIR \
                                        --guidance_scale GUIDANCE_SCALE \
                                        [--seed SEED] \
                                        [--class_ids CLASS_IDS [CLASS_IDS ...]] \
                                        [--respace_type RESPACE_TYPE] \
                                        [--respace_steps RESPACE_STEPS] \
                                        [--ddim] \
                                        [--ddim_eta DDIM_ETA] \
                                        [--batch_size BATCH_SIZE]
```

Basic arguments:

- `-c CONFIG`: path to the configuration file.
- `--weights WEIGHTS`: path to the model weights (checkpoint) file.
- `--sampler {ddpm,ddim}`: set the sampler.
- `--n_samples_each_class N_SAMPLES_EACH_CLASS`: number of samples for each class.
- `--save_dir SAVE_DIR`: path to the directory where samples will be saved.
- `--guidance_scale GUIDANCE_SCALE`: the guidance scale factor $s$
  - $s=0$: unconditional generation
  - $s=1$: non-guided conditional generation
  - $s>1$: guided conditional generation

Advanced arguments:

- `--class_ids CLASS_IDS [CLASS_IDS ...]`: a list of class ids to sample. If not specified, all classes will be sampled.
- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--batch_size BATCH_SIZE`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_cfg.py -h`.

For example, to sample 10 images for class (0, 2, 4, 8) from a pretrained CIFAR-10 model with guidance scale 3 using 100 DDIM steps:

```shell
accelerate-launch scripts/sample_cfg.py -c ./configs/ddpm_cfg_cifar10.yaml --weights /path/to/model/weights --sampler ddim --n_samples_each_class 10 --save_dir ./samples/cfg-cifar10 --guidance_scale 3 --class_ids 0 2 4 8 --respace_steps 100
```



## Evaluation

Sample 10K-50K images following the previous section and evaluate image quality with tools like [torch-fidelity](https://github.com/toshas/torch-fidelity), [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [clean-fid](https://github.com/GaParmar/clean-fid), etc.



## Results

**FID and IS on CIFAR-10 32x32**:

|       guidance scale       |  FID ↓   |      IS ↑       |
| :------------------------: | :------: | :-------------: |
|     0 (unconditional)      |  6.2904  | 8.9851 ± 0.0825 |
| 1 (non-guided conditional) |  4.6630  | 9.1763 ± 0.1201 |
|   3 (guided conditional)   | 10.2304  | 9.6252 ± 0.0977 |
|   5 (guided conditional)   | 16.23021 | 9.3210 ± 0.0744 |

- The images are sampled using DDIM with 50 steps.
- All the metrics are evaluated on 50K samples.
- FID measures diversity and IS measures fidelity. This table shows diversity-fidelity trade-off as guidance scale increases.



**Samples with different guidance scales on CIFAR-10 32x32**:

<p align="center">
  <img src="../assets/classifier-free-cifar10.png" />
</p>
From left to right: $s=0$ (unconditional), $s=1.0$ (non-guided conditional), $s=3.0$, $s=5.0$. Each row corresponds to a class.



**Samples with different guidance scales on ImageNet 256x256**:

The pretrained models are sourced from [openai/guided-diffusion](https://github.com/openai/guided-diffusion). Note that these models were initially designed for classifier guidance and thus are either conditional-only or unconditional-only. However, to facilitate classifier-free guidance, it would be more convenient if the model can handle both conditional and unconditional cases. To address this, I define a new class [UNetCombined](../models/adm/unet_combined.py), which combines the conditional-only and unconditional-only models into a single model. Also, we need to combine the pretrained weights for loading, which can be done by the following script:

```python
import yaml
from models.adm.unet_combined import UNetCombined


config_path = './weights/openai/guided-diffusion/256x256_diffusion.yaml'
weight_cond_path = './weights/openai/guided-diffusion/256x256_diffusion.pt'
weight_uncond_path = './weights/openai/guided-diffusion/256x256_diffusion_uncond.pt'
save_path = './weights/openai/guided-diffusion/256x256_diffusion_combined.pt'

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
model = UNetCombined(**cfg['model']['params'])
model.combine_weights(weight_cond_path, weight_uncond_path, save_path)
```



<p align="center">
  <img src="../assets/classifier-free-imagenet.png" />
</p>

From left to right: $s=1.0$ (non-guided conditional), $s=2.0$, $s=3.0$. Each row corresponds to a class.
