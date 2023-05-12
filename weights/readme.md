Besides training models by ourselves, this repo also supports loading other open source pretrained models and weights, as listed below. Each item is written in form `github_user_name/github_repo_name/xxx`, so you can visit the github page and find the way to download the pretrained models.

- ImageNet:
   - [only unconditional] `openai/guided-diffusion/256x256_diffusion_uncond.pt`
   - [only conditional] `openai/guided-diffusion/256x256_diffusion.pt`
- CelebA-HQ:
   - `andreas128/RePaint/celeba256_250000.pt`
   - `pesser/pytorch_diffusion/ema_diffusion_celebahq_model-560000.ckpt`
- LSUN-Church:
   - `pesser/pytorch_diffusion/ema_diffusion_lsun_church_model-4432000.ckpt`
- AFHQ-Dog:
   - `jychoi118/ilvr_adm/afhqdog_p2.pt`
- AFHQ-Cat:
   - `ChenWu98/cycle-diffusion/cat_ema_0.9999_050000.pt`
- AFHQ-Wild:
   - `ChenWu98/cycle-diffusion/wild_ema_0.9999_050000.pt`