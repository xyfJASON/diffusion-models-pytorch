# Classifier-Free Guidance

> Ho, Jonathan, and Tim Salimans. "Classifier-Free Diffusion Guidance." In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. 2021.



## Training

```shell
python train_classifier_free.py -c FILE -n NAME [--opts KEY1 VALUE1 KEY2 VALUE2 ...]
```

- To train on multiple GPUs, replace `python` with `torchrun --nproc_per_node NUM_GPUS`.
- An experiment directory will be created under `./runs/` for each run, which is named after `NAME`, or the current time if `NAME` is not specified. The directory contains logs, checkpoints, tensorboard, etc.

For example, to train on CIFAR-10:

```shell
python train_classifier_free.py -c ./configs/classifier_free_cifar10.yaml
```



## Sampling

```shell
python sample_classifier_free.py -c FILE \
                                 --model_path MODEL_PATH \
                                 [--load_ema] \
                                 --n_samples_each_class N_SAMPLES_EACH_CLASS \
                                 --guidance_scale GUIDANCE_SCALE \
                                 [--skip_steps SKIP_STEPS] \
                                 [--ddim] \
                                 [--ddim_eta DDIM_ETA] \
                                 --save_dir SAVE_DIR \
                                 [--batch_size BATCH_SIZE] 
                                 [--seed SEED] \
                                 [--opts KEY1 VALUE1 KEY2 VALUE2 ...]
```

- To sample on multiple GPUs, replace `python` with `torchrun --nproc_per_node NUM_GPUS`.
- Use `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps. 
- Use `--ddim` for DDIM sampling.
- Specify `--batch_size BATCH_SIZE` to sample images batch by batch. Set it as large as possible to fully utilize your devices. The default value of 1 is pretty slow.
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



**Qualitative results**:

<p align="center">
  <img src="../assets/classifier-free-cifar10.png" />
</p>


From left to right: $s=0$ (unconditional), $s=1.0$ (non-guided conditional), $s=3.0$, $s=5.0$. Each row corresponds to a class.

