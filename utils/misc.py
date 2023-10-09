import os
import sys
import tqdm
import torch
import shutil
import datetime
import importlib


def check_freq(freq: int, step: int):
    assert isinstance(freq, int)
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def image_float_to_uint8(image: torch.Tensor):
    """ [0, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert torch.ge(image, 0).all() and torch.le(image, 1).all()
    return (image * 255).to(dtype=torch.uint8)


def image_norm_to_float(image: torch.Tensor):
    """ [-1, 1] -> [0, 1] """
    assert image.dtype == torch.float32
    assert torch.ge(image, -1).all() and torch.le(image, 1).all()
    return (image + 1) / 2


def image_norm_to_uint8(image: torch.Tensor):
    """ [-1, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert torch.ge(image, -1).all() and torch.le(image, 1).all()
    return ((image + 1) / 2 * 255).to(dtype=torch.uint8)


def amortize(n_samples: int, batch_size: int):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def get_data_generator(dataloader, tqdm_kwargs):
    while True:
        for batch in tqdm.tqdm(dataloader, **tqdm_kwargs):
            yield batch


def find_resume_checkpoint(exp_dir: str, resume: str):
    """ Checkpoints are named after 'stepxxxxxx/' """
    if os.path.isdir(resume):
        ckpt_path = resume
    elif resume == 'best':
        ckpt_path = os.path.join(exp_dir, 'ckpt', 'best')
    elif resume == 'latest':
        d = dict()
        for name in os.listdir(os.path.join(exp_dir, 'ckpt')):
            if os.path.isdir(os.path.join(exp_dir, 'ckpt', name)) and name[:4] == 'step':
                d.update({int(name[4:]): name})
        ckpt_path = os.path.join(exp_dir, 'ckpt', d[sorted(d)[-1]])
    else:
        raise ValueError(f'resume option {resume} is invalid')
    assert os.path.isdir(ckpt_path), f'{ckpt_path} is not a directory'
    return ckpt_path


def instantiate_from_config(conf, **extra_params):
    module, cls = conf['target'].rsplit('.', 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**conf.get('params', dict()), **extra_params)


def create_exp_dir(
        exp_dir: str,
        conf_yaml: str,
        exist_ok: bool = False,
        time_str: str = None,
        no_interaction: bool = False,
):
    if time_str is None:
        time_str = get_time_str()
    if os.path.exists(exp_dir) and not exist_ok:
        cover = no_interaction or query_yes_no(
            question=f'{exp_dir} already exists! Cover it anyway?',
            default='no',
        )
        shutil.rmtree(exp_dir, ignore_errors=True) if cover else sys.exit(1)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    with open(os.path.join(exp_dir, f'config-{time_str}.yaml'), 'w') as f:
        f.write(conf_yaml)


def query_yes_no(question: str, default: str = "yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Copied from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
