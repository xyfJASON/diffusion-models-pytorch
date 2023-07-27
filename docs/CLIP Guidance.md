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
accelerate-launch sample_clip_guided.py -c CONFIG \
                                        [--seed SEED] \
                                        --weights WEIGHTS \
                                        [--var_type VAR_TYPE] \
                                        [--skip_type SKIP_TYPE] \
                                        [--skip_steps SKIP_STEPS] \
                                        --text TEXT \
                                        [--guidance_weight GUIDANCE_WEIGHT] \
                                        --n_samples N_SAMPLES \
                                        [--ddim] \
                                        [--ddim_eta DDIM_ETA] \
                                        --save_dir SAVE_DIR \
                                        [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Please wrap your text description with the quotation marks if it contains spaces, e.g., `--text 'a lovely dog'`.
- Use `--skip_type SKIP_TYPE` and `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.
- Specify `--micro_batch MICRO_BATCH` to sample images batch by batch. Set it as large as possible to fully utilize your devices.



## Results

**Pretrained on CelebA-HQ 256x256**:

<p align="center">
  <img src="../assets/clip-guidance-celebahq.png" width=80% />
</p>

All the images are sampled with 50 DDPM steps.

Images in the same row are sampled with the same random seed, thus share similar semantics. The first column shows the original samples (i.e., guidance scale=0). The next 3 columns are sampled using text description "a young girl with brown hair", with increasing guidance scales of 10, 50, and 100. The following 3 columns are similarly sampled with text description "an old man with a smile".

As expected, bigger guidance scale leads to greater changes. However, some results fail to match the descriptions, such as gender and hair color. Furthermore, guidance scale larger than 100 damages the image quality drastically.
