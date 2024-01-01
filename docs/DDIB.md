# DDIB

> Su, Xuan, Jiaming Song, Chenlin Meng, and Stefano Ermon. "Dual diffusion implicit bridges for image-to-image translation." *arXiv preprint arXiv:2203.08382* (2022).



## Sampling

```shell
accelerate-launch sample_ddib.py -c CONFIG \
                                 --weights WEIGHTS \
                                 --input_dir INPUT_DIR \
                                 --save_dir SAVE_DIR \
                                 --class_A CLASS_A \
                                 --class_B CLASS_B \
                                 [--seed SEED] \
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
- `--class_A CLASS_A`: input class label.
- `--class_B CLASS_B`: output class label.

Advanced arguments:
 
- `--respace_steps RESPACE_STEPS`: faster sampling that uses respaced timesteps.
- `--micro_batch MICRO_BATCH`: Batch size on each process. Sample by batch is faster, so set it as large as possible to fully utilize your devices.

See more details by running `python sample_ddib.py -h`.



## Results

**ImageNet 256x256 (conditional)** with pretrained model from [openai/guided-diffusion](https://github.com/openai/guided-diffusion):

<p align="center">
  <img src="../assets/ddib-imagenet.png" width=80% />
</p>

Notes: All images are sampled with 100 DDIM steps.

The results are not as good as expected. Some are acceptable, such as the Sussex Spaniel, Husky and Tiger in the 3rd row, but the others are not.
