import os
from abc import abstractmethod

import torch
import torch.distributed as dist
from numpy import inf

from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers.

    Saving behavior:
      - save_mode = "all": save periodic epoch checkpoints + save best checkpoint on improvement
      - save_mode = "best_only": save only best checkpoint on improvement
      - save_mode = "none": save nothing

    Monitor format in config:
      trainer.monitor = "min val_loss"  or  "max val_auc"  or  "off"
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, dataset, config, cv=None):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.dataset = dataset
        self.cv = cv

        # Distributed
        self.is_distributed = bool(config.get('distributed', False))
        self.dist_rank = dist.get_rank() if self.is_distributed else 0
        self.is_main = (self.dist_rank == 0)

        cfg_trainer = config['trainer']
        self.epochs = int(cfg_trainer['epochs'])
        self.save_period = int(cfg_trainer.get('save_period', 1))
        self.monitor = cfg_trainer.get('monitor', 'off')

        # Saving options
        self.save_mode = cfg_trainer.get('save_mode', 'all')  # 'all' | 'best_only' | 'none'
        assert self.save_mode in ('all', 'best_only', 'none'), f"Invalid save_mode: {self.save_mode}"

        self.save_best_on_improve = bool(cfg_trainer.get('save_best_on_improve', True))

        # Resume semantics for monitor_best:
        # - True: restore checkpoint['monitor_best'] (recommended)
        # - False: reset mnt_best to +/-inf after resume
        self.resume_restore_monitor_best = bool(cfg_trainer.get('resume_restore_monitor_best', True))

        # Monitoring / early stop
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_metric = None
            self.mnt_best = 0
            self.early_stop = inf
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ('min', 'max')
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop is None or self.early_stop <= 0:
                self.early_stop = inf

        # Epoch indexing (this trainer is explicitly 1-based)
        self.start_epoch = 1

        # Paths
        self.checkpoint_dir = config.save_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Tensorboard
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer.get('tensorboard', True))

        # Resume
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch: int):
        """
        Training logic for an epoch.

        Must return a dict of metrics, e.g.:
          {'loss': ..., 'val_loss': ..., 'val_auc': ...}
        """
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        last_log = None
        best_log = None

        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Build log dict
            log = {'epoch': epoch}
            if isinstance(result, dict):
                log.update(result)
            last_log = log

            # Print log
            for key, value in log.items():
                if isinstance(value, int):
                    self.logger.info('    {:15s}: {}'.format(str(key), value))
                else:
                    try:
                        self.logger.info('    {:15s}: {:3f}'.format(str(key), value))
                    except Exception:
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # Decide if improved
            best = False
            if self.mnt_mode != 'off':
                if self.mnt_metric not in log:
                    self.logger.warning(
                        "Warning: Metric '{}' not found in log. Monitoring disabled.".format(self.mnt_metric)
                    )
                    self.mnt_mode = 'off'
                else:
                    current = log[self.mnt_metric]
                    improved = (
                        (self.mnt_mode == 'min' and current <= self.mnt_best) or
                        (self.mnt_mode == 'max' and current >= self.mnt_best)
                    )

                    if improved:
                        self.mnt_best = current
                        not_improved_count = 0
                        best = True
                        best_log = log
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        self.logger.info(
                            "Validation performance didn't improve for {} epochs. Training stops."
                            .format(self.early_stop)
                        )
                        break

            # Saving: best immediately on improvement (never miss best)
            if self.is_main and best and self.save_best_on_improve and self.save_mode in ('all', 'best_only'):
                self._save_checkpoint(epoch, save_best=True, save_epoch_ckpt=(self.save_mode == 'all'))

            # Saving: periodic epoch checkpoints only in 'all' mode
            if self.is_main and self.save_mode == 'all':
                if self._should_save_epoch_ckpt(epoch):
                    self._save_checkpoint(epoch, save_best=False, save_epoch_ckpt=True)

        if self.is_distributed:
            dist.destroy_process_group()

        return best_log if best_log is not None else last_log

    def _should_save_epoch_ckpt(self, epoch: int) -> bool:
        """Periodic checkpoint schedule (only used when save_mode == 'all')."""
        if self.save_period <= 0:
            return False
        return (epoch % self.save_period == 0) or (epoch == self.epochs)

    def _checkpoint_filename(self, epoch: int) -> str:
        if self.cv is not None:
            return str(self.checkpoint_dir / f'fold{self.cv}-checkpoint-epoch{epoch}.pth')
        return str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')

    def _best_filename(self) -> str:
        if self.cv is not None:
            return str(self.checkpoint_dir / f'fold{self.cv}_model_best.pth')
        return str(self.checkpoint_dir / 'model_best.pth')

    def _save_checkpoint(self, epoch, save_best=False, save_epoch_ckpt=True):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        if save_epoch_ckpt:
            if self.cv is not None:
                filename = str(self.checkpoint_dir / f'fold{self.cv}-checkpoint-epoch{epoch}.pth')
            else:
                filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
            torch.save(state, filename)
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            if self.cv is not None:
                best_path = str(self.checkpoint_dir / f'fold{self.cv}_model_best.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best for current fold{self.cv}...")
            else:
                best_path = str(self.checkpoint_dir / 'model_best.pth')
                torch.save(state, best_path)
                self.logger.info("Saving current best: model_best.pth ...")


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoint.

        Note: uses torch.load(weights_only=False) per your preference.
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path, weights_only=False)

        self.start_epoch = int(checkpoint['epoch']) + 1

        # Restore monitor best (recommended) or reset (if you want "best since resume")
        if self.mnt_mode != 'off':
            if self.resume_restore_monitor_best and ('monitor_best' in checkpoint):
                self.mnt_best = checkpoint['monitor_best']
            else:
                self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        # Arch config warning
        if checkpoint.get('config', {}).get('arch', None) != self.config.get('arch', None):
            self.logger.warning(
                "Warning: Architecture config differs between checkpoint and current config."
            )

        # Load model
        state_dict = checkpoint['state_dict']
        if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        # Load optimizer if same type
        ckpt_opt_type = checkpoint.get('config', {}).get('optimizer', {}).get('type', None)
        cur_opt_type = self.config.get('optimizer', {}).get('type', None)
        if ckpt_opt_type != cur_opt_type:
            self.logger.warning(
                "Warning: Optimizer type differs between checkpoint and current config. "
                "Optimizer state will not be resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
