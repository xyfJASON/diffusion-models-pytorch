# CLIP Guidance

CLIP Guidance is a technique to generate images following an input text description with a pretrained diffusion model and a pretrained CLIP model. It uses CLIP score to guide the reverse diffusion process during sampling. To be specific, each reverse step is modified to:
$$
\begin{align}
&p_\theta(\mathbf x_{t-1}\vert\mathbf x_{t})=\mathcal N(\mathbf x_{t-1};\mu_\theta(\mathbf x_t,t){\color{dodgerblue}{+s\sigma_t^2\nabla_{\mathbf x_t}\mathcal L_{\text{CLIP}}}},\sigma_t^2\mathbf I)\\
&\mathcal L_\text{CLIP}=E_\text{image}(\mathbf x_\theta(\mathbf x_t,t))\cdot E_\text{text}(y)
\end{align}
$$
where $y$ is the input text, $E_\text{image}$ and $E_\text{text}$ are CLIP's image and text encoders, and $s$ is a hyper-parameter controlling the scale of guidance.



## Sampling

```shell
accelerate-launch sample_clip_guidance.py -c CONFIG \
                                          --weights WEIGHTS \
                                          --text TEXT \
                                          --n_samples N_SAMPLES \
                                          --save_dir SAVE_DIR \
                                          [--seed SEED] \
                                          [--var_type VAR_TYPE] \
                                          [--respace_type RESPACE_TYPE] \
                                          [--respace_steps RESPACE_STEPS] \
                                          [--guidance_weight GUIDANCE_WEIGHT] \
                                          [--clip_model CLIP_MODEL] \
                                          [--ddim] \
                                          [--ddim_eta DDIM_ETA] \
                                          [--micro_batch MICRO_BATCH]
```

This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.

Basic arguments:

- `-c CONFIG`: path to the configuration file.
- `--weights WEIGHTS`: path to the model weights (checkpoint) file.
- `--text TEXT`: text description. Please wrap your description with quotation marks if it contains spaces, e.g., `--text 'a lovely dog'`.
- `--n_samples N_SAMPLES`: number of samples to generate.
- `--save_dir SAVE_DIR`: path to the directory where samples will be saved.
- `--guidance_weight GUIDANCE_WEIGHT`: guidance weight (strength).
- `--clip_model CLIP_MODEL`: name of CLIP model. Default to "openai/clip-vit-base-patch32".

Advanced arguments:

- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--micro_batch MICRO_BATCH`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_clip_guidance -h`.



## Results

**Pretrained on CelebA-HQ 256x256**:

<p align="center">
  <img src="../assets/clip-guidance-celebahq.png" width=80% />
</p>

All the images are sampled with 50 DDPM steps.

Images in the same row are sampled with the same random seed, thus share similar semantics. The first column shows the original samples (i.e., guidance scale=0). The next 3 columns are sampled using text description "a young girl with brown hair", with increasing guidance scales of 10, 50, and 100. The following 3 columns are similarly sampled with text description "an old man with a smile".

As expected, bigger guidance scale leads to greater changes. However, some results fail to match the descriptions, such as gender and hair color. Furthermore, guidance scale larger than 100 damages the image quality drastically.
