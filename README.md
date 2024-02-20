# Diffusion-Models-Implementations

Implement Diffusion Models with PyTorch.

This is a **research-oriented** repository aiming to implement and reproduce diffusion models, including:

- [x] DDPM ([paper](https://arxiv.org/abs/2006.11239) | [website](https://hojonathanho.github.io/diffusion/) | [official repo](https://github.com/hojonathanho/diffusion))
- [x] DDIM ([paper](https://arxiv.org/abs/2010.02502) | [official repo](https://github.com/ermongroup/ddim))
- [x] Classifier-Free Guidance ([paper](https://arxiv.org/abs/2207.12598))
- [x] DDPM-IP ([paper](https://arxiv.org/abs/2301.11706) | [official repo](https://github.com/forever208/ddpm-ip))
- [x] CLIP Guidance
- [x] Mask Guidance ([RePaint paper](https://arxiv.org/abs/2201.09865) | [RePaint official repo](https://github.com/andreas128/RePaint))
- [x] ILVR ([paper](https://arxiv.org/abs/2108.02938) | [official repo](https://github.com/jychoi118/ilvr_adm))
- [x] SDEdit ([paper](https://arxiv.org/abs/2108.01073) | [website](https://sde-image-editing.github.io/) | [official repo](https://github.com/ermongroup/SDEdit))
- [x] DDIB ([paper](https://arxiv.org/abs/2203.08382) | [website](https://suxuann.github.io/ddib/) | [official repo](https://github.com/suxuann/ddib))
- [x] SD1.5 / 2.1 ([official repo](https://github.com/Stability-AI/stablediffusion))

<br/>



## Getting Started

### Environment

```shell
conda create -n diffusion python=3.11
conda activate diffusion
pip install -r requirements.txt
```



### Documentations

For instructions on training / sampling / evaluation, please refer to the [docs](./docs) folder.



### Pretrained weights

All the checkpoints and training logs trained by this repository are uploaded to [huggingface](https://huggingface.co/xyfJASON/Diffusion-Models-Implementations/tree/main).



### Loading models and weights from other repositories

Training a diffusion model on a large-scale dataset from scratch is time-consuming, especially with limited devices. Thus, this repository supports loading models and weights from other open source repositories, as listed below.

<table style="text-align: center">
    <tr>
        <th>Model Arch.</th>
        <th>Dataset</th>
        <th>Resolution</th>
        <th>Original Repo</th>
        <th>Config file</th>
    </tr>
    <tr>
        <td rowspan="2">UNet by pesser</td>
        <td>CelebA-HQ</td>
        <td>256x256</td>
        <td><a href="https://github.com/pesser/pytorch_diffusion">pesser/pytorch_diffusion</a></td>
        <td><a href="./configs/inference/pesser/pytorch_diffusion/ema_diffusion_celebahq_model-560000.yaml">config</a></td>
    </tr>
    <tr>
        <td>LSUN-Church</td>
        <td>256x256</td>
        <td><a href="https://github.com/pesser/pytorch_diffusion">pesser/pytorch_diffusion</a></td>
        <td><a href="./configs/inference/pesser/pytorch_diffusion/ema_diffusion_lsun_church_model-4432000.yaml">config</a></td>
    </tr>
    <tr>
        <td rowspan="4">ADM by openai</td>
        <td>ImageNet (unconditional)</td>
        <td>256x256</td>
        <td><a href="https://github.com/openai/guided-diffusion">openai/guided-diffusion</a></td>
        <td><a href="./configs/inference/openai/guided-diffusion/256x256_diffusion_uncond.yaml">config</a></td>
    </tr>
    <tr>
        <td>ImageNet (conditional)</td>
        <td>256x256</td>
        <td><a href="https://github.com/openai/guided-diffusion">openai/guided-diffusion</a></td>
        <td><a href="./configs/inference/openai/guided-diffusion/256x256_diffusion.yaml">config</a></td>
    </tr>
    <tr>
        <td>AFHQ-Dog</td>
        <td>256x256</td>
        <td><a href="https://github.com/jychoi118/ilvr_adm">jychoi118/ilvr_adm</a></td>
        <td><a href="./configs/inference/jychoi118/ilvr_adm/afhqdog_p2.yaml">config</a></td>
    </tr>
    <tr>
        <td>CelebA-HQ</td>
        <td>256x256</td>
        <td><a href="https://github.com/andreas128/RePaint">andreas128/RePaint</a></td>
        <td><a href="./configs/inference/andreas128/RePaint/celebahq_256_250000.yaml">config</a></td>
    </tr>
    <tr>
        <td rowspan="2">Stable Diffusion (v1.5 / v2.1)</td>
        <td>LAION</td>
        <td>512x512</td>
        <td><a href="https://github.com/runwayml/stable-diffusion">runwayml/stable-diffusion</a></td>
        <td><a href="./configs/inference/runwayml/stable-diffusion/v1-5-pruned-emaonly.yaml">config</a></td>
    </tr>
    <tr>
        <td>LAION</td>
        <td>768x768</td>
        <td><a href="https://github.com/Stability-AI/stablediffusion">Stability-AI/stablediffusion</a></td>
        <td><a href="./configs/inference/Stability-AI/stablediffusion/v2-1_768-ema-pruned.yaml">config</a></td>
    </tr>
</table>

The configuration files are located at `./configs/inference/<github username>/<repo name>/<weights filename>.yaml`, so it should be easy to find the corresponding weights.

<br/>



## Streamlit WebUI

Besides the command-line interface, this repo also provides a WebUI based on [Streamlit](https://streamlit.io/) library for easy interaction with the implemented models and algorithms. To run the WebUI, execute the following command:

```shell
streamlit run streamlit/Hello.py
```

<p align="center">
  <img src="./assets/streamlit.png" width=80% />
</p>

<br/>



## Preview

This section provides previews of the results generated by the implemented models and algorithms.

For more comprehensive quantitative and qualitative results, please refer to the documentations in the [docs](./docs) folder.



### DDPM

<p align="center">
  <img src="./assets/ddpm-mnist-random.png" width=30% />
  <img src="./assets/ddpm-cifar10-random.png" width=30% />
  <img src="./assets/ddpm-celebahq-random.png" width=30% />
</p>



### DDIM

<p align="center">
  <img src="./assets/ddim-cifar10.png" width=39% />
  <img src="./assets/ddim-cifar10-interpolate.png" width=50% />
</p>



### Classifier-Free Guidance

<p align="center">
  <img src="./assets/classifier-free-cifar10.png" width=90% />
</p>



### CLIP Guidance

<p align="center">
  <img src="./assets/clip-guidance-celebahq.png" width=80% />
</p>



### Mask Guidance

<p align="center">
  <img src="./assets/mask-guidance-imagenet.png" width=80% />
</p>



### ILVR

<p align="center">
  <img src="./assets/ilvr-celebahq.png" width=55% />
</p>



### SDEdit

<p align="center">
  <img src="./assets/sdedit.png" width=55% />
</p>



### DDIB

<p align="center">
  <img src="./assets/ddib-imagenet.png" width=80% />
</p>

<br/>



## Samplers: Fidelity-Speed Visualization

Once a diffusion model is trained, we can use different samplers to generate samples. The figure below shows the trade-off between fidelity and speed of different samplers, based on the same model trained on CIFAR-10 following the standard DDPM.

<p align="center">
  <img src="./assets/fidelity-speed-visualization.png" width=80% />
</p>

Notes:

- DDPM (fixed-small) is equivalent to DDIM(η=1).
- DDPM (fixed-large) performs better than DDPM (fixed-small) with 1000 steps, but degrades drastically as the number of steps decreases. If you check on the samples from DDPM (fixed-large) (<= 100 steps), you'll find that they still contain noticeable noises.

