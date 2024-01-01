import math
import torch


def get_beta_schedule(
        total_steps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
):
    """Get the beta schedule for diffusion.

    Args:
        total_steps: Number of diffusion steps.
        beta_schedule: Type of beta schedule. Options: 'linear', 'quad', 'const', 'cosine'.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        A Tensor of length `total_steps`.

    """
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float64)
    elif beta_schedule == 'quad':
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, total_steps, dtype=torch.float64) ** 2
    elif beta_schedule == 'const':
        return torch.full((total_steps, ), fill_value=beta_end, dtype=torch.float64)
    elif beta_schedule == 'cosine':
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = [
            min(1 - alpha_bar((i + 1) / total_steps) / alpha_bar(i / total_steps), 0.999)
            for i in range(total_steps)
        ]
        return torch.tensor(betas)
    else:
        raise ValueError(f'Beta schedule {beta_schedule} is not supported.')


def get_respaced_seq(
        total_steps: int = 1000,
        respace_type: str = 'uniform',
        respace_steps: int = 100,
):
    """Get respaced time sequence for fast inference.

    Args:
        total_steps: Number of the original diffusion steps.
        respace_type: Type of respaced timestep sequence. Options: 'uniform', 'uniform-leading', 'uniform-linspace',
         'uniform-trailing', 'quad', 'none', None.
        respace_steps: Length of respaced timestep sequence.

    Returns:
        A Tensor of length `respace_steps`, containing indices that are preserved in the respaced sequence.

    """
    if respace_type in ['uniform', 'uniform-leading']:
        space = total_steps // respace_steps
        seq = torch.arange(0, total_steps, space).long()
    elif respace_type == 'uniform-linspace':
        seq = torch.linspace(0, total_steps - 1, respace_steps).long()
    elif respace_type == 'uniform-trailing':
        space = total_steps // respace_steps
        seq = torch.arange(total_steps-1, -1, -space).long().flip(dims=[0])
    elif respace_type == 'quad':
        seq = torch.linspace(0, math.sqrt(total_steps * 0.8), respace_steps) ** 2
        seq = torch.floor(seq).long()
    elif respace_type is None or respace_type == 'none':
        seq = torch.arange(0, total_steps).long()
    else:
        raise ValueError(f'Respace type {respace_type} is not supported.')
    return seq


def _test_betas():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    betas_linear = get_beta_schedule(
        total_steps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
    )
    alphas_bar_linear = torch.cumprod(1. - betas_linear, dim=0)
    betas_quad = get_beta_schedule(
        total_steps=1000,
        beta_schedule='quad',
        beta_start=0.0001,
        beta_end=0.02,
    )
    alphas_bar_quad = torch.cumprod(1. - betas_quad, dim=0)
    betas_cosine = get_beta_schedule(
        total_steps=1000,
        beta_schedule='cosine',
        beta_start=0.0001,
        beta_end=0.02,
    )
    alphas_bar_cosine = torch.cumprod(1. - betas_cosine, dim=0)

    ax[0].plot(torch.arange(1000), betas_linear, label='linear')
    ax[0].plot(torch.arange(1000), betas_quad, label='quad')
    ax[0].plot(torch.arange(1000), betas_cosine, label='cosine')
    ax[0].set_title(r'$\beta_t$')
    ax[0].set_xlabel(r'$t$')
    ax[0].legend()
    ax[1].plot(torch.arange(1000), alphas_bar_linear, label='linear')
    ax[1].plot(torch.arange(1000), alphas_bar_quad, label='quad')
    ax[1].plot(torch.arange(1000), alphas_bar_cosine, label='cosine')
    ax[1].set_title(r'$\bar\alpha_t$')
    ax[1].set_xlabel(r'$t$')
    ax[1].legend()
    plt.show()


def _test_respace():
    seq = get_respaced_seq(
        total_steps=1000,
        respace_type='uniform-leading',
        respace_steps=10,
    )
    print('uniform-leading:\t', seq)
    seq = get_respaced_seq(
        total_steps=1000,
        respace_type='uniform-linspace',
        respace_steps=10,
    )
    print('uniform-linspace:\t', seq)
    seq = get_respaced_seq(
        total_steps=1000,
        respace_type='uniform-trailing',
        respace_steps=10,
    )
    print('uniform-trailing:\t', seq)


if __name__ == '__main__':
    _test_betas()
    # _test_respace()
