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
                                 [--var_type VAR_TYPE] \
                                 [--respace_type RESPACEP_TYPE] \
                                 [--respace_steps RESPACE_STEPS] \
                                 [--downsample_factor DOWNSAMPLE_FACTOR] \
                                 [--interp_method {cubic,lanczos2,lanczos3,linear,box}] \
                                 [--ddim] \
                                 [--ddim_eta DDIM_ETA] \
                                 [--micro_batch MICRO_BATCH]
```

This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.

Basic arguments:

- `-c CONFIG`: path to the configuration file.
- `--weights WEIGHTS`: path to the model weights (checkpoint) file.
- `--n_samples N_SAMPLES`: number of samples to generate.
- `--input_dir INPUT_DIR`: path to the directory where input images are saved.
- `--save_dir SAVE_DIR`: path to the directory where samples will be saved.
- `--downsample_factor DOWNSAMPLE_FACTOR`: higher factor leads to more diverse results.

Advanced arguments:

- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--micro_batch MICRO_BATCH`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_ilvr.py -h`.



## Notes

Using correct image resizing methods ([ResizeRight](https://github.com/assafshocher/ResizeRight)) is **CRUCIAL**! The default resizing functions in PyTorch (`torch.nn.functional.interpolate`) will damage the results.



## Results

**Pretrained CelebA-HQ 256x256**:

Note: All the images are sampled with 50 DDPM steps.

<p align="center">
  <img src="../assets/ilvr-celebahq.png" width=80% />
</p>
