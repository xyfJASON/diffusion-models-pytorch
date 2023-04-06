# DDPM

> Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.



## Training

```shell
accelerate-launch train_ddpm.py [-c CONFIG] [-e EXP_DIR] [--xxx.yyy zzz ...]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Results (logs, checkpoints, tensorboard, etc.) of each run will be saved to `EXP_DIR`. If `EXP_DIR` is not specified, they will be saved to `runs/exp-{current time}/`.
- To modify some configuration items without creating a new configuration file, you can pass `--key value` pairs to the script. For example, the default variance schedule in `./configs/ddpm_cifar10.yaml` is linear, and if you want to change it to cosine, you can simply pass `--diffusion.beta_schedule cosine`.

For example, to train on CIFAR-10 with default settings:

```shell
accelerate-launch train_ddpm.py -c ./configs/ddpm_cifar10.yaml
```



## Sampling

```shell
accelerate-launch sample_ddpm.py -c CONFIG \
                                 [--seed SEED] \
                                 --weights WEIGHTS \
                                 [--load_ema LOAD_EMA] \
                                 [--var_type VAR_TYPE] \
                                 [--skip_type SKIP_TYPE] \
                                 [--skip_steps SKIP_STEPS] \
                                 --n_samples N_SAMPLES \
                                 --save_dir SAVE_DIR \
                                 [--micro_batch MICRO_BATCH] \
                                 [--mode {sample,denoise,progressive}] \
                                 [--n_denoise N_DENOISE] \
                                 [--n_progressive N_PROGRESSIVE]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Use `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.
- Choose a sampling mode by `--mode MODE`, the options are:
  - `sample` (default): randomly sample images
  - `denoise`: sample images with visualization of its denoising process.
  - `progressive`:  sample images with visualization of its progressive generation process (i.e. predicted $x_0$).
- Specify `--micro_batch MICRO_BATCH` to sample images batch by batch. Set it as large as possible to fully utilize your devices.



## Evaluation

Sample 10K-50K images following the previous section and evaluate image quality with tools like [torch-fidelity](https://github.com/toshas/torch-fidelity), [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [clean-fid](https://github.com/GaParmar/clean-fid), etc.



## Results

**FID and IS on CIFAR-10 32x32**:

All the metrics are evaluated on 50K samples using [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/index.html) library.

<table align="center" width=100%>
  <tr>
    <th align="center">Type of variance</th>
    <th align="center">timesteps</th>
    <th align="center">FID ↓</th>
    <th align="center">IS ↑</th>
  </tr>
  <tr>
    <td align="center" rowspan="4">fixed-large</td>
    <td align="center">1000</td>
    <td align="center"><b>3.0459</b></td>
    <td align="center"><b>9.4515 ± 0.1179</b></td>
  </tr>
  <tr>
    <td align="center">100 (10x faster)</td>
    <td align="center">46.5454</td>
    <td align="center">8.7223 ± 0.0923</td>
  </tr>
  <tr>
    <td align="center">50 (20x faster)</td>
    <td align="center">85.2221</td>
    <td align="center">6.3863 ± 0.0894</td>
  </tr>
  <tr>
    <td align="center">10 (100x faster)</td>
    <td align="center">266.7540</td>
    <td align="center">1.5870 ± 0.0092</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">fixed-small</td>
    <td align="center">1000</td>
    <td align="center">5.3727</td>
    <td align="center">9.0118 ± 0.0968</td>
  </tr>
  <tr>
    <td align="center">100 (10x faster)</td>
    <td align="center">11.2191</td>
    <td align="center">8.6237 ± 0.0921</td>
  </tr>
  <tr>
    <td align="center">50 (20x faster)</td>
    <td align="center">15.0471</td>
    <td align="center">8.4077 ± 0.1623</td>
  </tr>
  <tr>
    <td align="center">10 (100x faster)</td>
    <td align="center">41.04793</td>
    <td align="center">7.1373 ± 0.0801</td>
  </tr>
 </table>



**Random samples**:

<p align="center">
  <img src="../assets/ddpm-mnist-random.png" width=30% />
  <img src="../assets/ddpm-cifar10-random.png" width=30% />
  <img src="../assets/ddpm-celebahq-random.png" width=30% />
</p>



**Denoising process**:

<p align="center">
  <img src="../assets/ddpm-cifar10-denoise.png" width=50% />
</p>
<p align="center">
  <img src="../assets/ddpm-celebahq-denoise.png" width=50% />
</p>



**Progressive generation**:

<p align="center">
  <img src="../assets/ddpm-cifar10-progressive.png" width=50% />
</p>
<p align="center">
  <img src="../assets/ddpm-celebahq-progressive.png" width=50% />
</p>
