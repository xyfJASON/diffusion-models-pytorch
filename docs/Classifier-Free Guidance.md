# Classifier-Free Guidance

> Ho, Jonathan, and Tim Salimans. "Classifier-Free Diffusion Guidance." In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. 2021.



## Training

```python
accelerate-launch train_classifier_free.py [-c CONFIG] [-e EXP_DIR] [--xxx.yyy zzz ...]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Results (logs, checkpoints, tensorboard, etc.) of each run will be saved to `EXP_DIR`. If `EXP_DIR` is not specified, they will be saved to `runs/exp-{current time}/`.

- To modify some configuration items without creating a new configuration file, you can pass `--key value` pairs to the script. For example, the default probability to disable guidance in training (`p_uncond`) in `./configs/classifier_free_cifar10.yaml` is 0.2, and if you want to change it to 0.1, you can simply pass `--train.p_uncond 0.1`.

For example, to train on CIFAR-10 with default settings:

```shell
accelerate-launch train_classifier_free.py -c ./configs/classifier_free_cifar10.yaml
```



## Sampling

```shell
accelerate-launch sample_classifier_free.py -c CONFIG \
                                            [--seed SEED] \
                                            --weights WEIGHTS \
                                            [--skip_type SKIP_TYPE] \
                                            [--skip_steps SKIP_STEPS] \
                                            --guidance_scale GUIDANCE_SCALE \
                                            --n_samples_each_class N_SAMPLES_EACH_CLASS \
                                            [--ddim] \
                                            [--ddim_eta DDIM_ETA] \
                                            --save_dir SAVE_DIR \
                                            [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Use `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps. 
- Use `--ddim` for DDIM sampling.
- Specify `--micro_batch MICRO_BATCH` to sample images batch by batch. Set it as large as possible to fully utilize your devices.
- I use $s$ in [Classifier Guidance paper](https://arxiv.org/abs/2105.05233) as the scale factor (`--guidance_scale`) rather than $w$ in the [Classifier-Free Guidance paper](https://arxiv.org/abs/2207.12598). In fact, we have $s=w+1$, and:
  - $s=0$: unconditional generation
  - $s=1$: non-guided conditional generation
  - $s>1$: guided conditional generation




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
import torch
from models.openai.guided_diffusion.unet_combined import UNetCombined


config_path = './configs/openai/guided-diffusion/256x256_diffusion.yaml'
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
