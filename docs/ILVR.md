# ILVR

Iterative Latent Variable Refinement (ILVR).

> Choi, Jooyoung, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. “ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models.” In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14347-14356. IEEE, 2021.



## Sampling

```shell
accelerate-launch sample_ilvr.py -c CONFIG \
                                 --weights WEIGHTS \
                                 --n_samples N_SAMPLES \
                                 --input_dir INPUT_DIR \
                                 --save_dir SAVE_DIR \
                                 [--seed SEED] \
                                 [--load_ema LOAD_EMA] \
                                 [--var_type VAR_TYPE] \
                                 [--skip_type SKIP_TYPE] \
                                 [--skip_steps SKIP_STEPS] \
                                 [--downsample_factor DOWNSAMPLE_FACTOR] \
                                 [--interp_method {cubic,lanczos2,lanczos3,linear,box}] \
                                 [--ddim] \
                                 [--ddim_eta DDIM_ETA] \
                                 [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Higher `--downsample_factor` leads to more diverse results.
  
- Use `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.



## Notes

Using correct image resizing methods ([ResizeRight](https://github.com/assafshocher/ResizeRight)) is **CRUCIAL**! The default resizing functions in PyTorch (`torch.nn.functional.interpolate`) will damage the results.



## Results

**Pretrained CelebA-HQ 256x256**:

Note: All the images are sampled with 50 DDPM steps.

<p align="center">
  <img src="../assets/ilvr-celebahq.png" width=80% />
</p>




