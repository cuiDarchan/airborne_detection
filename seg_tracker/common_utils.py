import time
from contextlib import contextmanager
import numpy as np
import warnings
import math

import torch
import yaml
import skimage.transform

import torch.utils.data
from typing import Sized, Optional

def argmax2d(x):
    return np.unravel_index(x.argmax(), x.shape)


def argmin2d(x):
    return np.unravel_index(x.argmin(), x.shape)


def build_geom_transform(
        dst_w,
        dst_h,
        src_center_x,
        src_center_y,
        scale_x=1.0,
        scale_y=1.0,
        angle=0.0,
        shear=0.0,
        hflip=False,
        vflip=False,
        return_params=False
):
    """
    @param dst_size_x:
    @param dst_size_y:
    @param src_center_x:
    @param src_center_y:
    @param scale_x:
    @param scale_y:
    @param angle:
    @param shear:
    @param hflip:
    @param vflip:
    @return: transform matrix

    Usage with cv2:

    crop = cv2.warpAffine(img, np.linalg.pinv(tform.params)[:2, :],
                          dsize=(self.frame_size, self.frame_size), flags=cv2.INTER_LINEAR)
    """

    if hflip:
        scale_x *= -1
    if vflip:
        scale_y *= -1

    tform = skimage.transform.AffineTransform(translation=(src_center_x, src_center_y))
    tform = skimage.transform.AffineTransform(scale=(1.0 / scale_x, 1.0 / scale_y)) + tform
    tform = skimage.transform.AffineTransform(rotation=angle * math.pi / 180,
                                              shear=shear * math.pi / 180) + tform
    tform = skimage.transform.AffineTransform(translation=(-dst_w / 2, -dst_h / 2)) + tform

    if return_params:
        return np.linalg.pinv(tform.params)
    else:
        return tform



def normalize_experiment_name(experiment_name: str):
    if experiment_name.startswith("experiments/"):
        experiment_name = experiment_name[len("experiments/") :]
    if experiment_name.endswith(".yaml"):
        experiment_name = experiment_name[: -len(".yaml")]

    return experiment_name


def load_config_data(experiment_name: str, experiments_dir='experiments') -> dict:
    with open(f"{experiments_dir}/{experiment_name}.yaml") as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print(f"[{name}] finished in {elapsedTime:0.3f}s")


def print_stats(title, array):
    if len(array):
        print(
            "{} shape:{} dtype:{} min:{} max:{} mean:{} median:{}".format(
                title,
                array.shape,
                array.dtype,
                np.min(array),
                np.max(array),
                np.mean(array),
                np.median(array),
            )
        )
    else:
        print(title, "empty")


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (float, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1.0, eta_min=0, last_epoch=-1, verbose=False, first_epoch_lr_scale=None):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1.0:
            raise ValueError("Expected T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.first_epoch_lr_scale = first_epoch_lr_scale

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lr_scale = 1.0
        if self.last_epoch == 0 and self.first_epoch_lr_scale is not None:
            lr_scale = self.first_epoch_lr_scale

        return [lr_scale * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = int(self.T_i * self.T_mult)
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1.0:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def check_CosineAnnealingWarmRestarts():
    import matplotlib.pyplot as plt

    optimizer = torch.optim.SGD([torch.tensor(1)], lr=1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1.42)

    lrs = []
    for _ in range(250):
        optimizer.step()
        lrs.append(scheduler.get_lr())
        scheduler.step()

    plt.plot(lrs)
    plt.show()


class RandomSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, num_samples: int) -> None:
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)

        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        for _ in range(self.num_samples // 32):
            yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    check_CosineAnnealingWarmRestarts()
    pass
