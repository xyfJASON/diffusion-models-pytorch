# DDPM-IP

> Ning, Mang, Enver Sangineto, Angelo Porrello, Simone Calderara, and Rita Cucchiara. "Input Perturbation Reduces Exposure Bias in Diffusion Models." *arXiv preprint arXiv:2301.11706* (2023).



## Training

This repo uses 🤗 Accelerate library for multi-GPUs/fp16 supports. Please read the documentation for how to launch the script on different platforms.

```shell
accelerate-launch scripts/train_ddpm_ip.py -c CONFIG [-e EXP_DIR] [--key value ...]
```

Arguments:

- `-c CONFIG`: path to the training configuration file.
- `-e EXP_DIR`: results (logs, checkpoints, tensorboard, etc.) will be saved to `EXP_DIR`. Default to `runs/exp-{current time}/`.
- `--key value`: modify configuration items in `CONFIG` via CLI.

For example, to train on CIFAR-10 with default settings:

```shell
accelerate-launch scripts/train_ddpm_ip.py -c ./configs/ddpm_ip_cifar10.yaml
```



## Sampling & Evaluation

Exactly the same as DDPM, refer to [doc](./DDPM.md) for more information.



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
    <td align="center">3.2497</td>
    <td align="center">9.4885 ± 0.09244</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">46.7994</td>
    <td align="center">8.5720 ± 0.0917</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">87.1883</td>
    <td align="center">6.1429 ± 0.0630</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">268.1108</td>
    <td align="center">1.5842 ± 0.0055</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">fixed-small</td>
    <td align="center">1000</td>
    <td align="center">4.4868</td>
    <td align="center">9.1092 ± 0.1025</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">9.2460</td>
    <td align="center">8.7068 ± 0.0813</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">12.7965</td>
    <td align="center">8.4902 ± 0.0701</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">35.5062</td>
    <td align="center">7.3680 ± 0.1092</td>
  </tr>
 </table>


The results are substantially better than DDPM for fixed-small variance, but not for fixed-large variance.

