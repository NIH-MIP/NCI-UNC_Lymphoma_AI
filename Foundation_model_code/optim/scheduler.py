import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from typing import Any, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


def build_scheduler(
    config: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    n_iter_per_epoch: int
) -> Any:
    """
    Build a learning rate scheduler based on config.

    Args:
        config: experiment configuration dict with 'trainer', 'optimizer', 'lr_scheduler' keys
        optimizer: Optimizer instance to wrap
        n_iter_per_epoch: number of update steps per epoch

    Returns:
        A scheduler instance (timm Scheduler or torch.optim.lr_scheduler)
    """
    epochs = config['trainer']['epochs']
    warmup_epochs = config['trainer'].get('warmup_epochs', 0)
    decay_epochs = config['lr_scheduler']['args'].get('decay_epochs', 0)

    total_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    decay_steps = int(decay_epochs * n_iter_per_epoch)

    sched_cfg = config['lr_scheduler']
    sched_type = sched_cfg['type'].lower()
    args = sched_cfg.get('args', {})
    opt_args = config['optimizer']['args']

    if sched_type == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=total_steps - warmup_steps if args.get('warmup_prefix', False) else total_steps,
            lr_min=opt_args.get('min_lr', 0.0),
            warmup_lr_init=opt_args.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            cycle_limit=5,
            t_in_epochs=False,
            warmup_prefix=args.get('warmup_prefix', False)
        )

    elif sched_type == 'linear':
        return LinearLRScheduler(
            optimizer,
            t_initial=total_steps,
            lr_min_rate=args.get('min_rate', 0.01),
            warmup_t=warmup_steps,
            warmup_lr_init=opt_args.get('warmup_lr', 0.0),
            t_in_epochs=False
        )

    elif sched_type == 'step':
        return StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=args.get('decay_rate', 0.1),
            warmup_lr_init=opt_args.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            t_in_epochs=False
        )
    elif sched_type == 'cosineannealinglr':
        T_max = args.get('T_max', epochs)
        eta_min = args.get('eta_min', opt_args.get('min_lr', 0.0))
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=-1
        )
    elif sched_type == 'cosine_restart':
        T_0 = args.get('T_0', epochs)
        eta_min = args.get('eta_min', opt_args.get('min_lr', 0.0))
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=1,
            eta_min=eta_min,
            last_epoch=-1
        )

    else:
        raise ValueError(f"Unknown scheduler type '{sched_cfg['type']}'")


class LinearLRScheduler(Scheduler):
    """
    Linearly decaying LR from base to lr_min_rate after warmup.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t: int = 0,
        warmup_lr_init: float = 0.0,
        t_in_epochs: bool = True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        # compute warmup increments per group
        if self.warmup_t > 0:
            self.warmup_steps = [
                (base - self.warmup_lr_init) / self.warmup_t for base in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [0.0 for _ in self.base_values]

    def _get_lr(self, t: int):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * step for step in self.warmup_steps]
        else:
            t_adj = t - self.warmup_t
            total_dec = self.t_initial - self.warmup_t
            return [
                base - (base - base * self.lr_min_rate) * (t_adj / total_dec)
                for base in self.base_values
            ]

    def get_epoch_values(self, epoch: int):
        return self._get_lr(epoch) if self.t_in_epochs else None

    def get_update_values(self, num_updates: int):
        return self._get_lr(num_updates) if not self.t_in_epochs else None
