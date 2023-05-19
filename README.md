# Diffusion-Models-Implementations

Implement Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [x] DDIM
- [x] Guidance
- [x] Classifier-Free Guidance

- Applications

  - [x] CLIP Guidance

  - [x] Mask Guidance

  - [x] ILVR
  - [x] SDEdit

<br/>



## Getting Started



### Instructions on training / sampling / evaluation

For instructions on training / sampling / evaluation, please refer to the [docs](./docs) folder.



### Loading models and weights from other repositories

Training a diffusion model on a large-scale dataset from scratch is time-consuming, especially with limited devices. Thus, this repository supports loading models and weights from other open source repositories, which are listed below.

<table style="text-align: center">
    <tr>
        <th>Model Architecture</th>
        <th>Dataset</th>
        <th>Original Repository</th>
        <th>Config file</th>
    </tr>
    <tr>
        <td rowspan="2">PyTorch diffusion models converted from TensorFlow by pesser</td>
        <td>CelebA-HQ (256x256)</td>
        <td><a href="https://github.com/pesser/pytorch_diffusion">pesser/pytorch_diffusion</a></td>
        <td><a href="./configs/pesser/pytorch_diffusion/celebahq.yaml">config</a></td>
    </tr>
    <tr>
        <td>LSUN-Church (256x256)</td>
        <td><a href="https://github.com/pesser/pytorch_diffusion">pesser/pytorch_diffusion</a></td>
        <td><a href="./configs/pesser/pytorch_diffusion/lsun_church.yaml">config</a></td>
    </tr>
    <tr>
        <td rowspan="4">ADM (guided diffusion models) by openai</td>
        <td>ImageNet (unconditional, 256x256)</td>
        <td><a href="https://github.com/openai/guided-diffusion">openai/guided-diffusion</a></td>
        <td><a href="./configs/openai/guided-diffusion/imagenet_256_uncond.yaml">config</a></td>
    </tr>
    <tr>
        <td>ImageNet (conditional, 256x256)</td>
        <td><a href="https://github.com/openai/guided-diffusion">openai/guided-diffusion</a></td>
        <td><a href="./configs/openai/guided-diffusion/imagenet_256_cond.yaml">config</a></td>
    </tr>
    <tr>
        <td>AFHQ-Dog (256x256)</td>
        <td><a href="https://github.com/jychoi118/ilvr_adm">jychoi118/ilvr_adm</a></td>
        <td><a href="./configs/openai/guided-diffusion/afhqdog_jychoi118_ilvr.yaml">config</a></td>
    </tr>
    <tr>
        <td>CelebA-HQ (256x256)</td>
        <td><a href="https://github.com/andreas128/RePaint">andreas128/RePaint</a></td>
        <td><a href="./configs/openai/guided-diffusion/celebahq_andreas128_RePaint.yaml">config</a></td>
    </tr>
</table>



<br/>



## Preview

This section provides a preview of the results achieved by the implemented methods and algorithms. For more comprehensive quantitative and qualitative results, please refer to the documentation in the [docs](./docs) folder.



### DDPM

[paper](https://arxiv.org/abs/2006.11239) | [website](https://hojonathanho.github.io/diffusion/) | [official repo](https://github.com/hojonathanho/diffusion)

<p align="center">
  <img src="./assets/ddpm-mnist-random.png" width=30% />
  <img src="./assets/ddpm-cifar10-random.png" width=30% />
  <img src="./assets/ddpm-celebahq-random.png" width=30% />
</p>


### DDIM

[paper](https://arxiv.org/abs/2010.02502) | [official repo](https://github.com/ermongroup/ddim)

<p align="center">
  <img src="./assets/ddim-cifar10.png" width=39% />
  <img src="./assets/ddim-cifar10-interpolate.png" width=50% />
</p>


### Classifier-Free Guidance

[paper](https://arxiv.org/abs/2207.12598)

<p align="center">
  <img src="./assets/classifier-free-cifar10.png" />
</p>


### CLIP Guidance

<p align="center">
  <img src="./assets/clip-guidance-celebahq.png" width=80% />
</p>


### Mask Guidance

[RePaint paper](https://arxiv.org/abs/2201.09865) | [RePaint official repo](https://github.com/andreas128/RePaint)

<p align="center">
  <img src="./assets/mask-guidance-imagenet.png" width=80% />
</p>



### ILVR

[paper](https://arxiv.org/abs/2108.02938) | [official repo](https://github.com/jychoi118/ilvr_adm)

<p align="center">
  <img src="./assets/ilvr-celebahq.png" width=55% />
</p>


### SDEdit

[paper](https://arxiv.org/abs/2108.01073) | [website](https://sde-image-editing.github.io/) | [official repo](https://github.com/ermongroup/SDEdit)

<p align="center">
  <img src="./assets/sdedit.png" width=55% />
</p>


<br/>



## Sampling Algorithms: Fidelity-Speed Visualization

I use the same model in all tests, which is trained following the standard DDPM. Thus the comparison depends only on the performance of different sampling algorithms (or SDE/ODE solvers).

<p align="center">
  <img src="./assets/fidelity-speed-visualization.png" width=80% />
</p>


Notes:

- DDPM (fixed-small) is equivalent to DDIM(η=1).
- DDPM (fixed-large) performs better than DDPM (fixed-small) with 1000 steps, but degrades drastically as the number of steps decreases. If you check on the samples from DDPM (fixed-large) (<= 100 steps), you'll find that they still contain noticeable noises.

