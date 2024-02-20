# Mask Guidance

Mask Guidance is a technique to fill the masked area in an input image with a pretrained diffusion model. It was first proposed in [1] and further developed in [2], [3], etc. for image inpainting.

Directly applying mask guidance may lead to inconsistent semantic between masked and unmasked areas. To overcome this problem, RePaint[3] proposed a resampling strategy, which goes forward and backward on the Markov chain from time to time.

> [1]. Song, Yang, and Stefano Ermon. “Generative modeling by estimating gradients of the data distribution.”
> Advances in neural information processing systems 32 (2019).
>
> [2]. Avrahami, Omri, Dani Lischinski, and Ohad Fried. “Blended diffusion for text-driven editing of natural
> images.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18208
> -18218. 2022.
>
> [3]. Lugmayr, Andreas, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. “Repaint:
> Inpainting using denoising diffusion probabilistic models.” In Proceedings of the IEEE/CVF Conference on
> Computer Vision and Pattern Recognition, pp. 11461-11471. 2022.



## Sampling

This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the script on different platforms.

```shell
accelerate-launch scripts/sample_mask_guidance.py  -c CONFIG \
                                                   --weights WEIGHTS \
                                                   --input_dir INPUT_DIR \
                                                   --save_dir SAVE_DIR \
                                                   [--seed SEED] \
                                                   [--var_type VAR_TYPE] \
                                                   [--respace_type RESPACE_TYPE] \
                                                   [--respace_steps RESPACE_STEPS] \
                                                   [--resample] \
                                                   [--resample_r RESAMPLE_R] \
                                                   [--resample_j RESAMPLE_J] \
                                                   [--batch_size BATCH_SIZE]
```

Basic arguments:

- `-c CONFIG`: path to the configuration file.
- `--weights WEIGHTS`: path to the model weights (checkpoint) file.
- `--input_dir INPUT_DIR`: path to the directory where input images are saved.
- `--save_dir SAVE_DIR`: path to the directory where samples will be saved.
- `--resample`: use the resample strategy proposed in RePaint paper[3]. This strategy has two hyperparameters:
  - `--resample_r RESAMPLE_R`: number of resampling.
  - `--resample_j RESAMPLE_J`: jump lengths.

Advanced arguments:

- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--batch_size BATCH_SIZE`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_mask_guidance.py -h`.



## Results

**ImageNet 256x256** with pretrained model from [openai/guided-diffusion](https://github.com/openai/guided-diffusion):

<p align="center">
  <img src="../assets/mask-guidance-imagenet.png" width=80% />
</p>

Notes:

- All the images are sampled with 50 DDPM steps.
- Jump length $j$ is fixed to 10.
- $r=1$ is equivalent to the original DDPM sampling (w/o resampling).

