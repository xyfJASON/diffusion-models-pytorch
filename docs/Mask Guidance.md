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

```shell
accelerate-launch sample_mask_guided.py -c CONFIG \
                                        [--seed SEED] \
                                        --weights WEIGHTS \
                                        [--load_ema LOAD_EMA] \
                                        [--var_type VAR_TYPE] \
                                        [--skip_type SKIP_TYPE] \
                                        [--skip_steps SKIP_STEPS] \
                                        [--resample] \
                                        [--resample_r RESAMPLE_R] \
                                        [--resample_j RESAMPLE_J] \
                                        --n_samples N_SAMPLES \
                                        --save_dir SAVE_DIR \
                                        [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Use `--resample` for resample strategy as proposed in RePaint paper[3]. This strategy has two hyper-parameters:
  - `--resample_r RESAMPLE_R`: number of resampling.
  - `--resample_j RESAMPLE_J`: jump lengths.

- Use `--skip_type SKIP_TYPE` and `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.



## Results

**ImageNet 256x256** with pretrained model from [openai/guided-diffusion](https://github.com/openai/guided-diffusion):

<p align="center">
  <img src="../assets/mask-guidance-imagenet.png" width=80% />
</p>

Notes:

- All the images are sampled with 50 DDPM steps.
- Jump length $j$ is fixed to 10.
- $r=1$ is equivalent to the original DDPM sampling (w/o resampling).

