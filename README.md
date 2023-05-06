# Diffusion-Models-Implementations

My implementations of Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [x] DDIM
- [x] Classifier-Free Guidance
- [ ] Guidance
  - [x] CLIP Guidance
  - [x] Mask Guidance
  - [x] ILVR


<br/>



## Instructions and More Results

For instructions on training / sampling / evaluation and more quantitative & qualitative results, please refer to [docs](./docs).

<br/>



## DDPM \[<a href="https://arxiv.org/abs/2006.11239">paper</a>\] \[<a href="https://hojonathanho.github.io/diffusion/">website</a>\] \[<a href="https://github.com/hojonathanho/diffusion">official repo</a>\]

<p align="center">
  <img src="./assets/ddpm-mnist-random.png" width=30% />
  <img src="./assets/ddpm-cifar10-random.png" width=30% />
  <img src="./assets/ddpm-celebahq-random.png" width=30% />
</p>
<br/>



## DDIM \[<a href="https://arxiv.org/abs/2010.02502">paper</a>\] \[<a href="https://github.com/ermongroup/ddim">official repo</a>\]

<p align="center">
  <img src="./assets/ddim-cifar10.png" width=39% />
  <img src="./assets/ddim-cifar10-interpolate.png" width=50% />
</p>
<br/>



## Classifier-Free Guidance \[<a href="https://arxiv.org/abs/2207.12598">paper</a>\]

<p align="center">
  <img src="./assets/classifier-free-cifar10.png" />
</p>
<br/>



## CLIP Guidance

<p align="center">
  <img src="./assets/clip-guidance-celebahq.png" width=80% />
</p>
<br/>



## Mask Guidance

<p align="center">
  <img src="./assets/mask-guidance-imagenet.png" width=80% />
</p>

<br/>



## ILVR

<p align="center">
  <img src="./assets/ilvr-celebahq.png" width=55% />
</p>


<br/>



## Sampling Algorithms: Fidelity-Speed Visualization

I use the same model in all tests, which is trained following the standard DDPM. Thus the comparison depends only on the performance of different sampling algorithms (or SDE/ODE solvers).

<p align="center">
  <img src="./assets/fidelity-speed-visualization.png" width=80% />
</p>


Interesting facts observed:

- DDPM (fixed-large) performs better than DDPM (fixed-small) with 1000 steps, but degrades drastically as the number of steps decreases. If you check on the samples from DDPM (fixed-large) (<= 100 steps), you'll find that they still contain noticeable noises.

