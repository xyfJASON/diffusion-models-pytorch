# SDEdit

> Meng, Chenlin, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. "Sdedit: Guided image synthesis and editing with stochastic differential equations." In *International Conference on Learning Representations*. 2021.



## Sampling

```shell
accelerate-launch sample_sdedit.py -c CONFIG \
                                   --weights WEIGHTS \
                                   --input_dir INPUT_DIR \
                                   --save_dir SAVE_DIR \
                                   --edit_steps EDIT_STEPS \
                                   [--seed SEED] \
                                   [--var_type VAR_TYPE] \
                                   [--respace_type RESPACE_TYPE] \
                                   [--respace_steps RESPACE_STEPS] \
                                   [--micro_batch MICRO_BATCH]
```

This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.

Basic arguments:

- `-c CONFIG`: path to the configuration file.
- `--weights WEIGHTS`: path to the model weights (checkpoint) file.
- `--input_dir INPUT_DIR`: path to the directory where input images are saved.
- `--save_dir SAVE_DIR`: path to the directory where samples will be saved.
- `--edit_steps EDIT_STEPS`: number of edit steps. Controls realism-faithfulness trade-off.

Advanced arguments:

- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--micro_batch MICRO_BATCH`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_sdedit.py -h`.



## Results

**LSUN-Church 256x256** with pretrained model from [pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion):

<p align="center">
  <img src="../assets/sdedit.png" width=80% />
</p>
