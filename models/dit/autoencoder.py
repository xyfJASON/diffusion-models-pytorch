import diffusers


def AutoEncoderKL(from_pretrained: str):
    return diffusers.AutoencoderKL.from_pretrained(from_pretrained)
