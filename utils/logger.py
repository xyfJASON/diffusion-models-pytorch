import os
import tqdm
import logging
from typing import Dict

from utils.dist import master_only, get_dist_info


class AverageMeter:
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MessageLogger:
    def __init__(self, log_root: str = None, print_freq: int = 0):
        assert isinstance(print_freq, int) and print_freq >= 0
        self.print_freq = print_freq
        # logging
        if log_root is not None:
            self.logger = get_logger(log_file=os.path.join(log_root, 'output.log'), file_mode='a')
        else:
            self.logger = get_logger()
        # tensorboard
        if log_root is not None:
            self.tb_logger = get_tb_writer(log_dir=os.path.join(log_root, 'tensorboard'))
        else:
            self.tb_logger = None

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def close(self):
        if self.tb_logger:
            self.tb_logger.close()

    def track_status(self, name: str, status: Dict, global_step: int, epoch: int, iteration: int = None):
        message = f'[{name}] epoch: {epoch}' + ('' if iteration is None else f', iteration: {iteration}')
        for k, v in status.items():
            message += f', {k}: {v:.6f}'
            if self.tb_logger:
                self.tb_logger.add_scalar(f'{name}/{k}', v, global_step)
        if self.print_freq and (global_step + 1) % self.print_freq == 0:
            self.logger.info(message)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)


def get_logger(name='exp', log_file=None, log_level=logging.INFO, file_mode='w'):
    """ Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/logging.py """
    logger = logging.getLogger(name)
    # Check if the logger exists
    if logger.hasHandlers():
        return logger
    # Add a stream handler
    # stream_handler = logging.StreamHandler()
    stream_handler = TqdmLoggingHandler()
    handlers = [stream_handler]
    # Add a file handler for master process (global rank == 0)
    global_rank = get_dist_info()['global_rank']
    if global_rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    # Set format & level for all handlers
    # Note that levels of non-master processes are always 'ERROR'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level if global_rank == 0 else logging.ERROR)
        logger.addHandler(handler)
    logger.setLevel(log_level if global_rank == 0 else logging.ERROR)
    return logger


@master_only
def get_tb_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
    return tb_writer
