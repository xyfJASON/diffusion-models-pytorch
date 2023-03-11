import os
import tqdm
import logging
from typing import Dict, List
from torch import Tensor


def get_logger(name: str = 'exp',
               log_file: str = None,
               log_level: int = logging.INFO,
               file_mode: str = 'w',
               use_tqdm_handler: bool = False,
               is_main_process: bool = True):
    logger = logging.getLogger(name)
    # Check if the logger exists
    if logger.hasHandlers():
        return logger
    # Add a stream handler
    if not use_tqdm_handler:
        stream_handler = logging.StreamHandler()
    else:
        stream_handler = TqdmLoggingHandler()
    handlers = [stream_handler]
    # Add a file handler for main process
    if is_main_process and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    # Set format & level for all handlers
    # Note that levels of non-master processes are always 'ERROR'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_level = log_level if is_main_process else logging.ERROR
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)


class StatusTracker:
    def __init__(
            self,
            logger: logging.Logger,
            exp_dir: str,
            print_freq: int = 0,
            is_main_process: bool = True,
    ):
        self.logger = logger
        self.print_freq = print_freq
        self.tb_writer = None
        if is_main_process:
            self.tb_writer = get_tb_writer(log_dir=os.path.join(exp_dir, 'tensorboard'))

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()

    def track_status(self, name: str, status: Dict, step: int, write_tb: List[bool] = None):
        message = f'[{name}] step: {step}'
        for i, (k, v) in enumerate(status.items()):
            if isinstance(v, Tensor):
                v = v.item()
            message += f', {k}: {v:.6f}'
            if self.tb_writer is not None and (write_tb is None or write_tb[i] is True):
                self.tb_writer.add_scalar(f'{name}/{k}', v, step)
        if self.print_freq > 0 and (step + 1) % self.print_freq == 0:
            self.logger.info(message)


def get_tb_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir)
    return tb_writer
