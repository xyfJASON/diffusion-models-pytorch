import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import glob
import logging
from omegaconf import OmegaConf

import torch
import numpy as np
import streamlit as st

from utils.load import load_weights
from utils.misc import instantiate_from_config, image_norm_to_uint8

logpy = logging.getLogger("models.sdxl.attention")
logpy.setLevel(logging.ERROR)


WEIGHTS_PREFIX = "weights/sdxl"


@st.cache_resource
def build_model(conf_model, weights_path, low_vram=False):
    build_model.clear()
    torch.cuda.empty_cache()
    assert conf_model["target"] == "models.sdxl.stablediffusion.StableDiffusion"
    model = instantiate_from_config(conf_model, low_vram_shift_enabled=low_vram)
    weights = load_weights(os.path.join(WEIGHTS_PREFIX, weights_path))
    model.load_state_dict(weights)
    return model


@st.cache_resource
def build_diffuser(conf_diffusion, sampler, device, respace_type, respace_steps, cfg_scale):
    if sampler == "DDPM":
        conf_diffusion["target"] = "diffusions.DDPMCFG"
    elif sampler == "DDIM":
        conf_diffusion["target"] = "diffusions.DDIMCFG"
    diffuser = instantiate_from_config(
        conf_diffusion,
        cond_kwarg="condition_dict",
        respace_type=None if respace_steps is None else respace_type,
        respace_steps=respace_steps,
        guidance_scale=cfg_scale,
        device=device,
    )
    return diffuser


def main(
        st_components, conf, weights_path, seed, sampler, respace_type, respace_steps, offset_noise,
        pos_prompt, neg_prompt, height, width, cfg_scale, batch_size, batch_count, low_vram,
):
    # SYSTEM SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # BUILD DIFFUSER
    conf_diffusion = OmegaConf.to_container(conf.diffusion)
    diffuser = build_diffuser(conf_diffusion, sampler, device, respace_type, respace_steps, cfg_scale)

    # BUILD MODEL & LOAD WEIGHTS
    conf_model = OmegaConf.to_container(conf.model)
    model = build_model(conf_model, weights_path, low_vram)
    model.to(device).eval()

    # START SAMPLING
    start_time = time.time()
    sample_list = []
    for i in range(batch_count):
        with st_components["placeholder_image"]:
            st.write(f"Generating images... {i}/{batch_count}")
        with torch.no_grad():
            img_shape = (4, height // 8, width // 8)
            init_noise = torch.randn((batch_size, *img_shape), device=device)
            if offset_noise > 0.0:
                noise = offset_noise * torch.randn((init_noise.shape[0], ), device=device)
                init_noise = init_noise + noise[..., None, None, None]
            cond_dict = model.conditioner_forward(text=[pos_prompt] * batch_size, H=height, W=width)
            uncond_dict = model.conditioner_forward(text=[neg_prompt] * batch_size, H=height, W=width)
            samples = diffuser.sample(
                model=model, init_noise=init_noise,
                uncond_conditioning=uncond_dict,
                model_kwargs=dict(condition_dict=cond_dict),
                tqdm_kwargs=dict(desc=f'Fold {i}/{batch_count}'),
            )
            samples = model.decode_latent(samples).clamp(-1, 1)
            samples = image_norm_to_uint8(samples)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
        sample_list.extend([s for s in samples])
    end_time = time.time()
    with st_components["placeholder_image"]:
        st.image(sample_list, output_format="PNG")
    with st_components["container_image_meta"]:
        st.text(f"Seed: {seed}    Time taken: {end_time - start_time:.2f} seconds")
    torch.cuda.empty_cache()


def streamlit():
    # STREAMLIT SETUP
    st.set_page_config(page_title="Diffusion", layout="wide")
    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 450px;
           max-width: 450px;
       }
       """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("pageid", None) != "Stable Diffusion XL":
        st.cache_resource.clear()
        torch.cuda.empty_cache()
    st.session_state.pageid = "Stable Diffusion XL"

    # PAGE TITLE
    st.title("Stable Diffusion XL")

    # CONFIG PATH
    config_path = os.path.join(WEIGHTS_PREFIX, "sd_xl_base.yaml")
    conf = OmegaConf.load(config_path)

    cols = st.columns(2)
    with cols[0]:
        # MODEL SELECTION
        container_model = st.container(border=False)
        with container_model:
            extensions = ["pt", "pth", "ckpt", "safetensors"]
            weights_list = []
            for ext in extensions:
                weights_list.extend(glob.glob(os.path.join(WEIGHTS_PREFIX, f"**/*.{ext}"), recursive=True))
            weights_list = [w[len(WEIGHTS_PREFIX)+1:] for w in sorted(weights_list)]
            weights_path = st.selectbox("Model", options=weights_list, index=None)
        # PROMPT BOX
        with st.container(border=True):
            pos_prompt = st.text_area("Prompt", value="A photo of a cat", height=200)
            neg_prompt = st.text_area(
                "Negative prompt", height=200,
                value="lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality",
            )

    with cols[1]:
        # BUTTON
        cols[1].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        bttn_generate = st.button(
            "Generate", use_container_width=True, type="primary",
            disabled=config_path is None or weights_path is None,
        )
        # IMAGE DISPLAY
        container_image_meta = st.container(border=True)
        with container_image_meta:
            st.markdown("Output")
            placeholder_image = st.empty()

    with st.sidebar:
        # BASIC OPTIONS
        expander_basic_options = st.expander("Basic options", expanded=True)
        with expander_basic_options:
            seed = st.number_input(f"Seed", min_value=-1, max_value=2**32-1, value=-1, step=1)
            if seed == -1:
                seed = np.random.randint(0, 2**32-1)

            cols = st.columns(2)
            with cols[0]:
                sampler = st.selectbox("Sampler", options=["DDPM", "DDIM"])
            with cols[1]:
                max_value = conf.diffusion.params.total_steps
                respace_steps = st.number_input("Sample steps", min_value=1, max_value=max_value, value=20)

            cols = st.columns(2)
            with cols[0]:
                height = st.select_slider("Image height", options=range(512, 2048+1, 128), value=1024)
            with cols[1]:
                width = st.select_slider("Image width", options=range(512, 2048+1, 128), value=1024)

            cfg_scale = st.slider("CFG scale", min_value=1.0, max_value=20.0, value=7.0, step=0.1)

            cols = st.columns(2)
            with cols[0]:
                batch_size = st.number_input("Batch size", min_value=1, value=1)
            with cols[1]:
                batch_count = st.number_input("Batch count", min_value=1, value=1)

        # ADVANCED OPTIONS
        expander_advanced_options = st.expander("Advanced options", expanded=False)
        with expander_advanced_options:
            respace_type = st.selectbox("Respace type", options=["uniform-linspace", "uniform-leading", "uniform-trailing"])
            offset_noise = st.slider("Offset noise", min_value=0.0, max_value=0.1, value=0.0, step=0.01)
            low_vram = st.checkbox("Low vram")

    # GENERATE IMAGES
    if bttn_generate:
        main(
            st_components=dict(
                container_model=container_model,
                expander_basic_options=expander_basic_options,
                container_image_meta=container_image_meta,
                placeholder_image=placeholder_image,
            ),
            conf=conf,
            weights_path=weights_path,
            seed=seed,
            sampler=sampler,
            respace_type=respace_type,
            respace_steps=respace_steps,
            offset_noise=offset_noise,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            height=height,
            width=width,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            batch_count=batch_count,
            low_vram=low_vram,
        )


if __name__ == "__main__":
    streamlit()
