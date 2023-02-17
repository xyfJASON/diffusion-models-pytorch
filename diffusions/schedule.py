import math
import torch


def get_beta_schedule(beta_schedule: str = 'linear',
                      total_steps: int = 1000,
                      beta_start: float = 0.0001,
                      beta_end: float = 0.02):
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float64)
    elif beta_schedule == 'quad':
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, total_steps, dtype=torch.float64) ** 2
    elif beta_schedule == 'const':
        return torch.full((total_steps, ), fill_value=beta_end, dtype=torch.float64)
    elif beta_schedule == "cosine":
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = [
            min(1 - alpha_bar((i + 1) / total_steps) / alpha_bar(i / total_steps), 0.999)
            for i in range(total_steps)
        ]
        return torch.tensor(betas)
    else:
        raise ValueError(f'Beta schedule {beta_schedule} is not supported.')


def get_skip_seq(skip_type: str = 'uniform', skip_steps: int = 1000, total_steps: int = 1000):
    if skip_type == 'uniform':
        skip = total_steps // skip_steps
        seq = torch.arange(0, total_steps, skip)
    elif skip_type == 'quad':
        seq = torch.linspace(0, math.sqrt(total_steps * 0.8), skip_steps) ** 2
        seq = torch.floor(seq).to(dtype=torch.int64)
    else:
        raise ValueError(f'skip_type {skip_type} is not valid')
    return seq


def _test_skip():
    seq = get_skip_seq(
        skip_type='uniform',
        skip_steps=10,
        total_steps=1000,
    )
    print(seq)


def _test_betas():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    betas_linear = get_beta_schedule('linear', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_linear = torch.cumprod(1. - betas_linear, dim=0)
    betas_quad = get_beta_schedule('quad', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_quad = torch.cumprod(1. - betas_quad, dim=0)
    betas_cosine = get_beta_schedule('cosine', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_cosine = torch.cumprod(1. - betas_cosine, dim=0)
    ax[0].plot(torch.arange(1000), betas_linear, label='linear')
    ax[0].plot(torch.arange(1000), betas_quad, label='quad')
    ax[0].plot(torch.arange(1000), betas_cosine, label='cosine')
    ax[0].set_title('betas')
    ax[0].legend()
    ax[1].plot(torch.arange(1000), alphas_bar_linear, label='linear')
    ax[1].plot(torch.arange(1000), alphas_bar_quad, label='quad')
    ax[1].plot(torch.arange(1000), alphas_bar_cosine, label='cosine')
    ax[1].set_title('alphas_bar')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    _test_betas()
