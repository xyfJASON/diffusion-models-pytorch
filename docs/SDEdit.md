# SDEdit

> Meng, Chenlin, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. "Sdedit: Guided image synthesis and editing with stochastic differential equations." In *International Conference on Learning Representations*. 2021.



## Sampling

```shell
accelerate-launch sample_sdedit.py -c CONFIG [--seed SEED] \
                                   --weights WEIGHTS \
                                   [--load_ema LOAD_EMA] \
                                   [--var_type VAR_TYPE] \
                                   [--skip_type SKIP_TYPE] \
                                   [--skip_steps SKIP_STEPS] \
                                   --edit_steps EDIT_STEPS \
                                   --input_dir INPUT_DIR \
                                   --save_dir SAVE_DIR \
                                   [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- `--edit_steps EDIT_STEPS` controls realism-faithfulness trade-off.
  
- Use `--skip_type SKIP_TYPE` and `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.



## Results

**LSUN-Church 256x256** with pretrained model from [pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion):

<p align="center">
  <img src="../assets/sdedit.png" width=80% />
</p>




