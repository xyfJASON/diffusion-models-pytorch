# Repro-Diffusion-Models

Reproduce Diffusion Models with PyTorch.



## Progress

- [x] DDPM
- [ ] DDIM



## Docs

Things about how to run the code.

- [DDPM](./docs/DDPM.md)



## Results



### DDPM

**Quantitative results:**

<table width=100%>
  <tr>
    <th align="center">Dataset</th>
    <th align="center">notes</th>
    <th align="center">FID</th>
    <th align="center">IS</th>
  </tr>
  <tr>
    <th align="center" rowspan="5">CIFAR-10 (32x32)</th>
    <td align="center">fixed_large</td>
    <td align="center">3.1246</td>
    <td align="center">9.3690 (0.1015)</td>
  </tr>
  <tr>
    <td align="center">fixed_small</td>
    <td align="center">5.3026</td>
    <td align="center">8.9711 (0.1172)</td>
  </tr>
  <tr>
    <td align="center">100 timesteps (10x faster)<br/>fixed_small</td>
    <td align="center">11.1331</td>
    <td align="center">8.5436 (0.1291)</td>
  </tr>
  <tr>
    <td align="center">50 timesteps (20x faster)<br/>fixed_small</td>
    <td align="center">15.5682</td>
    <td align="center">8.3658 (0.0665)</td>
  </tr>
  <tr>
    <td align="center">10 timesteps (100x faster)<br/>fixed_small</td>
    <td align="center">40.8977</td>
    <td align="center"> 7.1148 (0.0824)</td>
  </tr>
 </table>


**Qualitative results**:

<table width=100%>
  <tr>
    <th width=10% align="center">Dataset</th>
    <th width=40% align="center">MNIST</th>
    <th width=40% align="center">CIFAR-10</th>
  </tr>
  <tr>
    <th align="center">Random samples</th>
    <td align="center"><img src="./assets/ddpm-mnist-random.png"/></td>
    <td align="center"><img src="./assets/ddpm-cifar10-random.png"/></td>
  </tr>
  <tr>
    <th align="center">Denoising process</th>
    <td align="center"><img src="./assets/ddpm-mnist-denoise.png"/></td>
    <td align="center"><img src="./assets/ddpm-cifar10-denoise.png"/></td>
  </tr>
 </table>
