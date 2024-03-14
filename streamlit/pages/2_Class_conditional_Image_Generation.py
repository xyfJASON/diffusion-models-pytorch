import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import glob
from omegaconf import OmegaConf

import torch
import numpy as np
import streamlit as st

from models.base_latent import BaseLatent
from utils.load import load_weights
from utils.misc import instantiate_from_config, image_norm_to_uint8


WEIGHTS_PREFIX = "weights"

AVAILABLE_WEIGHTS = [
    "facebookresearch/DiT",
    "openai/guided-diffusion/256x256_diffusion_combined",
    "sail-sg/MDT",
    "xyfJASON",
]


def check_is_available(path):
    for weight in AVAILABLE_WEIGHTS:
        if weight in path:
            return True
    return False


@st.cache_resource
def build_model(conf_model, weights_path):
    build_model.clear()
    torch.cuda.empty_cache()
    model = instantiate_from_config(conf_model)
    weights = load_weights(weights_path)
    model.load_state_dict(weights)
    return model


@st.cache_resource
def build_diffuser(conf_diffusion, sampler, device, var_type, respace_type, respace_steps, cfg_scale):
    if sampler == "DDPM":
        conf_diffusion["target"] = "diffusions.DDPMCFG"
    elif sampler == "DDIM":
        conf_diffusion["target"] = "diffusions.DDIMCFG"
    diffuser = instantiate_from_config(
        conf_diffusion,
        var_type=var_type or conf_diffusion["params"].get("var_type", None),
        respace_type=None if respace_steps is None else respace_type,
        respace_steps=respace_steps,
        guidance_scale=cfg_scale,
        cond_kwarg="y",
        device=device,
    )
    return diffuser


def main(
        st_components, conf, weights_path, seed, sampler, class_label,
        cfg_scale, respace_steps, batch_size, batch_count, var_type, respace_type,
):
    # SYSTEM SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # BUILD DIFFUSER
    conf_diffusion = OmegaConf.to_container(conf.diffusion)
    diffuser = build_diffuser(conf_diffusion, sampler, device, var_type, respace_type, respace_steps, cfg_scale)

    # BUILD MODEL & LOAD WEIGHTS
    conf_model = OmegaConf.to_container(conf.model)
    model = build_model(conf_model, weights_path)
    model.to(device).eval()
    is_latent = isinstance(model, BaseLatent)

    # START SAMPLING
    start_time = time.time()
    sample_list = []
    for i in range(batch_count):
        with st_components["placeholder_image"]:
            st.write(f"Generating images... {i}/{batch_count}")
        with torch.no_grad():
            if is_latent:
                img_shape = (4, conf.data.params.img_size // 8, conf.data.params.img_size // 8)
            else:
                img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
            init_noise = torch.randn((batch_size, *img_shape), device=device)
            y = torch.full((batch_size, ), fill_value=class_label, dtype=torch.long, device=device)
            samples = diffuser.sample(
                model=model, init_noise=init_noise,
                model_kwargs=dict(y=y),
                tqdm_kwargs=dict(desc=f'Fold {i}/{batch_count}'),
            )
            if is_latent:
                samples = model.decode_latent(samples).clamp(-1, 1)
            else:
                samples = samples.clamp(-1, 1)
        samples = image_norm_to_uint8(samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()
        sample_list.extend([s for s in samples])
    end_time = time.time()
    with st_components["placeholder_image"]:
        st.image(sample_list, output_format="PNG")
    st_components["container_image_meta"].text(f"Seed: {seed}    Time taken: {end_time - start_time:.2f} seconds")
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
    if st.session_state.get("pageid", None) != "Class-conditional Image Generation":
        st.cache_resource.clear()
        torch.cuda.empty_cache()
    st.session_state.pageid = "Class-conditional Image Generation"

    # PAGE TITLE
    st.title("Class-conditional Image Generation")

    # MODEL SELECTION
    cols = st.columns([3, 1, 1])
    with cols[0]:
        extensions = ["pt", "pth", "ckpt", "safetensors"]
        weights_list = []
        for ext in extensions:
            weights_list.extend(glob.glob(os.path.join(WEIGHTS_PREFIX, f"**/*.{ext}"), recursive=True))
        weights_list = [w[len(WEIGHTS_PREFIX)+1:] for w in sorted(weights_list)]
        weights_list = filter(check_is_available, weights_list)
        weights_path = st.selectbox("Model", options=weights_list, index=None)
        weights_path = os.path.join(WEIGHTS_PREFIX, weights_path) if weights_path else None

    # LOAD CONFIG
    conf = None
    if weights_path is not None:
        config_path = os.path.splitext(weights_path)[0] + ".yaml"
        conf = OmegaConf.load(config_path)

    # CLASS SELECTION
    with cols[1]:
        max_value = conf.data.num_classes - 1 if conf else 999
        class_label = st.number_input(f"Class label (0~{max_value})", min_value=0, max_value=max_value, value=0, step=1)

    # BUTTON
    with cols[2]:
        cols[2].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        bttn_generate = st.button("Generate", use_container_width=True, type="primary", disabled=weights_path is None)

    # IMAGE DISPLAY
    container_image_meta = st.container(border=True)
    with container_image_meta:
        st.markdown("Output")
        placeholder_image = st.empty()

    with st.sidebar:
        # BASIC OPTIONS
        with st.expander("Basic options", expanded=True):
            seed = st.number_input("Seed", min_value=-1, max_value=2**32-1, value=-1, step=1)
            if seed == -1:
                seed = np.random.randint(0, 2**32-1)

            cols = st.columns(2)
            with cols[0]:
                sampler = st.selectbox("Sampler", options=["DDPM", "DDIM"])
            with cols[1]:
                max_value = conf.diffusion.params.total_steps if conf else 1000
                respace_steps = st.number_input("Sample steps", min_value=1, max_value=max_value, value=50)

            cfg_scale = st.slider("CFG scale", min_value=1.0, max_value=20.0, value=3.0, step=0.5)

            cols = st.columns(2)
            with cols[0]:
                batch_size = st.number_input("Batch size", min_value=1, value=1)
            with cols[1]:
                batch_count = st.number_input("Batch count", min_value=1, value=1)

        # ADVANCED OPTIONS
        with st.expander("Advanced options"):
            options = ["fixed_small", "fixed_large", "learned_range"]
            if conf:
                options.insert(0, options.pop(options.index(conf.diffusion.params.var_type)))
            var_type = st.selectbox("Type of variance", options=options)

            respace_type = st.selectbox("Respace type", options=["uniform-linspace", "uniform-leading", "uniform-trailing"])

    # GENERATE IMAGES
    if bttn_generate:
        main(
            st_components=dict(
                container_image_meta=container_image_meta,
                placeholder_image=placeholder_image,
            ),
            conf=conf,
            weights_path=weights_path,
            seed=seed,
            sampler=sampler,
            respace_steps=respace_steps,
            class_label=class_label,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            batch_count=batch_count,
            var_type=var_type,
            respace_type=respace_type,
        )


if __name__ == "__main__":
    streamlit()
