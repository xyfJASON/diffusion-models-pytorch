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
                                            [--load_ema LOAD_EMA] \
                                            [--skip_steps SKIP_STEPS] \
                                            --n_samples_each_class N_SAMPLES_EACH_CLASS \
                                            --guidance_scale GUIDANCE_SCALE \
                                            [--ddim] \
                                            [--ddim_eta DDIM_ETA] \
                                            --save_dir SAVE_DIR [--micro_batch MICRO_BATCH]
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

Same as DDPM. Please refer to [DDPM doc](./DDPM.md).



## Results

**FID and IS on CIFAR-10 32x32**:

<table align="center" width=100%>
  <tr>
    <th align="center">guidance scale</th>
    <th align="center">FID ↓</th>
    <th align="center">IS ↑</th>
  </tr>
  <tr>
    <td align="center">0 (unconditional)</td>
    <td align="center">6.1983</td>
    <td align="center">8.9323 (0.1542)</td>
  </tr>
  <tr>
    <td align="center">1 (non-guided conditional)</td>
    <td align="center">4.6546</td>
    <td align="center">9.2524 (0.1606)</td>
  </tr>
  <tr>
    <td align="center">3 (unconditional)</td>
    <td align="center">9.9375</td>
    <td align="center">9.5522 (0.1013)</td>
  </tr>
  <tr>
    <td align="center">5 (unconditional)</td>
    <td align="center">13.3187</td>
    <td align="center">9.4688 (0.1588)</td>
  </tr>
</table>


- The images are sampled using DDIM with 50 steps.
- All the metrics are evaluated on 50K samples.
- FID measures diversity and IS measures fidelity. This table shows diversity-fidelity trade-off as guidance scale increases.



**Samples with different guidance scale**:

<p align="center">
  <img src="../assets/classifier-free-cifar10.png" />
</p>


From left to right: $s=0$ (unconditional), $s=1.0$ (non-guided conditional), $s=3.0$, $s=5.0$. Each row corresponds to a class.

