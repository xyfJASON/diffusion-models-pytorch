import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import glob
from omegaconf import OmegaConf

import torch
import numpy as np
import streamlit as st

from utils.load import load_weights
from utils.misc import instantiate_from_config, image_norm_to_uint8


@st.cache_resource
def build_model(conf_model, weights_path):
    model = instantiate_from_config(conf_model)
    weights = load_weights(weights_path)
    model.load_state_dict(weights)
    return model


@st.cache_resource
def build_diffuser(conf_diffusion, sampler, device, var_type, respace_steps):
    if sampler == 'DDPM':
        conf_diffusion["target"] = "diffusions.ddpm.DDPM"
    elif sampler == 'DDIM':
        conf_diffusion["target"] = "diffusions.ddim.DDIM"
    diffuser = instantiate_from_config(
        conf_diffusion,
        var_type=var_type or conf_diffusion["params"].get("var_type", None),
        respace_type=None if respace_steps is None else "uniform",
        respace_steps=respace_steps,
        device=device,
    )
    return diffuser


# noinspection PyUnusedLocal
def main(
        st_components, config_path, weights_path, seed, sampler,
        respace_steps, batch_size, batch_count, var_type,
):
    # SYSTEM SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # LOAD CONFIG
    conf = OmegaConf.load(config_path)

    # BUILD DIFFUSER
    conf_diffusion = OmegaConf.to_container(conf.diffusion)
    diffuser = build_diffuser(conf_diffusion, sampler, device, var_type, respace_steps)

    # BUILD MODEL & LOAD WEIGHTS
    try:
        conf_model = OmegaConf.to_container(conf.model)
        model = build_model(conf_model, weights_path)
    except RuntimeError:
        st_components["container_config_model"].error(
            "Failed to load model weights. Please check if "
            "the config file and weights file are compatible."
        )
        st.stop()
    model.to(device).eval()

    # START SAMPLING
    start_time = time.time()
    sample_list = []
    for i in range(batch_count):
        with st_components["placeholder_image"]:
            st.write(f"Generating images... {i}/{batch_count}")
        with torch.no_grad():
            img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
            init_noise = torch.randn((batch_size, *img_shape), device=device)
            samples = diffuser.sample(
                model=model, init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{batch_count}'),
            ).clamp(-1, 1)
        samples = image_norm_to_uint8(samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()
        sample_list.extend([s for s in samples])
    end_time = time.time()
    with st_components["placeholder_image"]:
        st.image(sample_list, output_format="PNG")
    st_components["container_image_meta"].text(f"Seed: {seed}    Time taken: {end_time - start_time:.2f} seconds")


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

    # PAGE TITLE
    st.title("Unconditional Image Generation")

    # CONFIG & MODEL SELECTION
    OK = True
    col_left, col_right = st.columns([8, 1])
    with col_left:
        container_config_model = st.container(border=True)
        with container_config_model:
            cols = st.columns(2)
            with cols[0]:
                config_list = glob.glob(os.path.join("configs", "inference", "**/*.yaml"), recursive=True)
                config_list = sorted(config_list)
                config_path = st.selectbox("Config file", options=config_list, index=None)
                if config_path is None:
                    OK = False
            with cols[1]:
                extensions = ["pt", "pth", "ckpt", "safetensors"]
                weights_list = []
                for ext in extensions:
                    weights_list.extend(glob.glob(os.path.join("weights", f"**/*.{ext}"), recursive=True))
                weights_list = sorted(weights_list)
                weights_path = st.selectbox("Model", options=weights_list, index=None)
                if weights_path is None:
                    OK = False

    # BUTTON
    with col_right:
        bttn_generate = st.button("Generate", disabled=not OK, use_container_width=True, type="primary")

    with st.sidebar:
        # BASIC OPTIONS
        expander_basic_options = st.expander("Basic options", expanded=True)
        with expander_basic_options:
            cols = st.columns(2)
            with cols[0]:
                seed = st.number_input("Seed", min_value=-1, max_value=2**32-1, value=-1, step=1)
                if seed == -1:
                    seed = np.random.randint(0, 2**32-1)
            with cols[1]:
                sampler = st.selectbox("Sampler", options=["DDPM", "DDIM"])

            step_options = list(range(1, 20, 1)) + list(range(20, 100, 5)) + list(range(100, 1001, 50))
            respace_steps = st.select_slider("Sample steps", options=step_options, value=50)

            cols = st.columns(2)
            with cols[0]:
                batch_size = st.number_input("Batch size", min_value=1, value=1)
            with cols[1]:
                batch_count = st.number_input("Batch count", min_value=1, value=1)

        # ADVANCED OPTIONS
        expander_advanced_options = st.expander("Advanced options")
        with expander_advanced_options:
            var_type = st.selectbox("Type of variance", options=["default", "fixed_small", "fixed_large", "learned_range"])
            if var_type == "default":
                var_type = None

    # IMAGE DISPLAY
    container_image_meta = st.container(border=True)
    with container_image_meta:
        st.markdown("Output")
        placeholder_image = st.empty()

    # GENERATE IMAGES
    if bttn_generate:
        st_components = dict(
            container_config_model=container_config_model,
            expander_advanced_options=expander_advanced_options,
            expander_basic_options=expander_basic_options,
            container_image_meta=container_image_meta,
            placeholder_image=placeholder_image,
        )
        main(
            st_components=st_components,
            config_path=config_path,
            weights_path=weights_path,
            seed=seed,
            sampler=sampler,
            respace_steps=respace_steps,
            batch_size=batch_size,
            batch_count=batch_count,
            var_type=var_type,
        )


if __name__ == "__main__":
    streamlit()
