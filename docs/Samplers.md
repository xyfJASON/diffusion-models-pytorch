# Samplers: Fidelity-Speed Visualization

Once a diffusion model is trained, we can use different samplers (SDE / ODE solvers) to generate samples. Currently, this repo supports the following samplers:

- DDPM
- DDIM
- Euler

We can choose the number of steps for the samplers to generate samples. Generally speaking, the more steps we use, the better the fidelity of the samples. However, the speed of the sampler also decreases as the number of steps increases. Therefore, it is important to choose the right number of steps to balance the trade-off between fidelity and speed.

The table and figure below show the trade-off between fidelity and speed of different samplers, based on the same model trained on CIFAR-10 following the standard DDPM. All the metrics are evaluated on 50K samples using [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/index.html) library.

<table align="center" width=100%>
  <tr>
    <th align="center">sampler</th>
    <th align="center">NFE</th>
    <th align="center">FID ↓</th>
    <th align="center">IS ↑</th>
  </tr>
  <tr>
    <td align="center" rowspan="4">DDPM (fixed-large)</td>
    <td align="center">1000</td>
    <td align="center"><b>3.0459</b></td>
    <td align="center"><b>9.4515 ± 0.1179</b></td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">46.5454</td>
    <td align="center">8.7223 ± 0.0923</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">85.2221</td>
    <td align="center">6.3863 ± 0.0894</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">266.7540</td>
    <td align="center">1.5870 ± 0.0092</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">DDPM (fixed-small)</td>
    <td align="center">1000</td>
    <td align="center">5.3727</td>
    <td align="center">9.0118 ± 0.0968</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">11.2191</td>
    <td align="center">8.6237 ± 0.0921</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">15.0471</td>
    <td align="center">8.4077 ± 0.1623</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">41.0479</td>
    <td align="center">7.1373 ± 0.0801</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">DDIM (eta=0)</td>
    <td align="center">1000</td>
    <td align="center">4.1892</td>
    <td align="center">9.0626 ± 0.1093</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">6.0508</td>
    <td align="center">8.8424 ± 0.0862</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">7.7011</td>
    <td align="center">8.7076 ± 0.1021</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">18.9559</td>
    <td align="center">8.0852 ± 0.1137</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Euler</td>
    <td align="center">1000</td>
    <td align="center">4.2099</td>
    <td align="center">9.0678 ± 0.1191</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">6.0469</td>
    <td align="center">8.8511 ± 0.1054</td>
  </tr>
  <tr>
    <td align="center">50</td>
    <td align="center">7.6770</td>
    <td align="center">8.7217 ± 0.1122</td>
  </tr>
  <tr>
    <td align="center">10</td>
    <td align="center">18.7698</td>
    <td align="center">8.0287 ± 0.0781</td>
  </tr>
 </table>



<p align="center">
  <img src="../assets/fidelity-speed-visualization.png" width=100% />
</p>



Notes:

- DDPM (fixed-small) is equivalent to DDIM(η=1).
- DDPM (fixed-large) performs better than DDPM (fixed-small) with 1000 steps, but degrades drastically as the number of steps decreases. If you check on the samples from DDPM (fixed-large) (<= 100 steps), you'll find that they still contain noticeable noises.
- Euler sampler and DDIM (η=0) have almost the same performance.
