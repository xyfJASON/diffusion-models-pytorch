# DDIB

> Su, Xuan, Jiaming Song, Chenlin Meng, and Stefano Ermon. "Dual diffusion implicit bridges for image-to-image translation." *arXiv preprint arXiv:2203.08382* (2022).



## Sampling

```shell
accelerate-launch sample_sdedit.py -c CONFIG \
                                   [--seed SEED] \
                                   --input_dir INPUT_DIR \
                                   --save_dir SAVE_DIR \
                                   --weights WEIGHTS \
                                   --class_A CLASS_A \
                                   --class_B CLASS_B \
                                   [--skip_type SKIP_TYPE] \
                                   [--skip_steps SKIP_STEPS] \
                                   [--micro_batch MICRO_BATCH]
```

- This repo uses the [🤗 Accelerate](https://huggingface.co/docs/accelerate/index) library for multi-GPUs/fp16 supports. Please read the [documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) on how to launch the scripts on different platforms.
- Use `--class_A` and `--class_B` to specify the input class label and output class label.
  
- Use `--skip_type SKIP_TYPE` and `--skip_steps SKIP_STEPS` for faster sampling that skip timesteps.



## Results

**ImageNet 256x256 (conditional)** with pretrained model from [openai/guided-diffusion](https://github.com/openai/guided-diffusion):

<p align="center">
  <img src="../assets/ddib-imagenet.png" width=80% />
</p>

Notes: All images are sampled with 100 DDIM steps.

The results are not as good as expected. Some are acceptable, such as the Sussex Spaniel, Husky and Tiger in the 3rd row, but the others are not.
