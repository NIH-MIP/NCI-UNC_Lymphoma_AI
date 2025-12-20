import os
import time
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from base import BaseTrainer
import model.metric as module_metric
from model.loss_util.custom_loss import SupConLoss
from timm.utils import AverageMeter
from utils import inf_loop, MetricTracker, NativeScalerWithGradNormCount


class Trainer(BaseTrainer):
    """
    Trainer class for MIL models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Any,
        metric_ftns: List[Any],
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        data_loader: Any,
        dataset: Any,
        valid_data_loader: Optional[Any] = None,
        test_data_loader: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        local_rank: Optional[int] = None,
        len_epoch: Optional[int] = None,
        cv: Optional[int] = None,
    ) -> None:
        super().__init__(model, criterion, metric_ftns, optimizer, dataset, config)
        self.config = config
        self.device = device
        self.cv = cv
        self.mixed_precision = config['trainer']['mixed_precision']
        self.accumulation_steps = config['trainer']['accumulate_steps']
        self.clip_grad = config['trainer']['clip_grad']
        self.total_epoch = config['trainer']['epochs']
        self.supcon_criterion = SupConLoss(temperature=0.07)

        # Model metadata
        self.model_name = getattr(model, 'module', model).name
        self.model_conf = getattr(getattr(model, 'module', model), 'conf', None)

        # Data loader setup
        if len_epoch is None:
            self.len_epoch = len(data_loader)
            self.data_loader = data_loader
        else:
            self.len_epoch = len_epoch
            self.data_loader = inf_loop(data_loader)

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        # Logging intervals
        self.log_step = max(1, len(self.data_loader) // 2)

        # Metrics and scaler
        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in metric_ftns], writer=self.writer
        )
        self.loss_scaler = NativeScalerWithGradNormCount()

        self.local_rank = local_rank
        self.log_dir = Path(config['log_dir'])

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.train_metrics.reset()
        self.model.train()
        self.optimizer.zero_grad()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        outputs_list, targets_list = [], []

        ctx = torch.cuda.amp.autocast if self.mixed_precision else contextlib.nullcontext
        start = time.time()

        for batch_idx, batch in enumerate(self.data_loader):
            data = batch['image'].to(self.device)
            coords = batch['coords'].to(self.device)
            targets = batch['label'].to(self.device)
            features_tme = batch.get('features_tme')
            
            if features_tme is not None:
                features_tme = features_tme.to(self.device)

            with ctx(enabled=self.mixed_precision, dtype=torch.bfloat16):
                outputs, loss = self._forward_and_loss(
                    self.model,
                    self.model_name,
                    data,
                    coords,
                    targets,
                    features_tme,
                    self.criterion,
                    self.config,
                    self.supcon_criterion,
                )
            loss = loss / self.accumulation_steps + 1e-10

            is_second = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            grad_norm = self.loss_scaler(
                loss,
                self.optimizer,
                clip_grad=self.clip_grad,
                parameters=self.model.parameters(),
                create_graph=is_second,
                update_grad=(batch_idx + 1) % self.accumulation_steps == 0,
            )

            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == self.len_epoch:
                self.optimizer.zero_grad()
                if hasattr(self.lr_scheduler, 'step_update'):
                    self.lr_scheduler.step_update((epoch * self.len_epoch + batch_idx) //self.accumulation_steps)
                elif isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    pass  # will be stepped after validation
                elif hasattr(self.lr_scheduler, 'step'):
                    self.lr_scheduler.step()

            torch.cuda.synchronize()

            batch_time.update(time.time() - start)
            loss_meter.update(loss.item(), data.size(0))

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            outputs_list.append(outputs)
            targets_list.append(targets)

            if batch_idx % self.log_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                memory_used = torch.cuda.max_memory_allocated(device=self.device) / (1024.0 * 1024.0)
                self.logger.info(
                    f"Train Epoch: {epoch}/{self.total_epoch} "
                    f"[{batch_idx}/{self.len_epoch}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"LR: {lr:.4e} "
                    f"VRAM: {memory_used:.0f}MB"
                )
            if batch_idx == self.len_epoch:
                break

        all_outputs = torch.cat(outputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        self.train_metrics.update('loss', loss_meter.avg)
        for m in self.metric_ftns:
            self.train_metrics.update(m.__name__, m(all_outputs, all_targets, params=self.config))
        train_log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            train_log.update(**{'val_' + k: v for k, v in val_log.items()})
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(val_log['val_auc'])

        return train_log


    @torch.no_grad()
    def _valid_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.valid_metrics.reset()
        loss_meter = AverageMeter()

        outputs_list, targets_list = [], []

        for batch_idx, batch in enumerate(self.valid_data_loader):
            data = batch['image'].to(self.device)
            coords = batch['coords'].to(self.device)
            targets = batch['label'].to(self.device)

            features_tme = batch.get('features_tme')
            if features_tme is not None:
                features_tme = features_tme.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                outputs, loss = self._forward_and_loss(
                    self.model,
                    self.model_name,
                    data,
                    coords,
                    targets,
                    features_tme,
                    self.criterion,
                    self.config,
                    self.supcon_criterion,
                )

            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            loss_meter.update(loss.item(), data.size(0))
            outputs_list.append(outputs)
            targets_list.append(targets)

        all_outputs = torch.cat(outputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        self.valid_metrics.update('loss', loss_meter.avg)

        for m in self.metric_ftns:
            self.valid_metrics.update(m.__name__, m(all_outputs, all_targets, params=self.config))
        return self.valid_metrics.result()


    @torch.no_grad()
    def evaluate(self, mode: str = 'valid', verbose: bool = True):
        """
        Evaluate the best‐model checkpoint for this fold, matching the per‐epoch metrics
        but with an optimal Youden threshold for binary classification.
        Returns a 11‑tuple:
          (auc, precision, sensitivity, specificity,
           accuracy, f1,
           patient_ids, threshold,
           probabilities, predictions, truths)
        """
        
        # Pick loader
        if mode == 'valid':
            eval_loader = self.valid_data_loader
        elif mode == 'test':
            eval_loader = self.test_data_loader
        else:
            raise ValueError(f"No data loader for mode={mode}")

        # Load best checkpoint
        ckpt_name = f"fold{self.cv}_model_best.pth"
        ckpt_path = self.checkpoint_dir / ckpt_name
        
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        checkpoint = self._safe_torch_load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()
        self.logger.info(f"Validate the model: {ckpt_path}")

        # Run through the data
        patient_id = []
        slide_id = []
        outputs_list, targets_list = [], []
        t_ratio = []
        t_patch_count = []

        for batch in eval_loader:
            pid = batch['patient_id']
            sid = batch['slide_id']
            data   = batch['image'].to(self.device)
            features_tme = batch.get('features_tme')
            if features_tme is not None:
                features_tme = features_tme.to(self.device)
            coords = batch['coords'].to(self.device)
            targets = batch['label'].to(self.device)
            tumor_ratio = batch['tumor_ratio']
            tumor_patch_count = batch['tumor_patch_count']

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                # forward + ignore loss
                outputs, _ = self._forward_and_loss(
                    self.model,
                    self.model_name,
                    data,
                    coords,
                    targets,
                    features_tme,
                    self.criterion,
                    self.config,
                    self.supcon_criterion,
                )

            patient_id.append(pid)
            slide_id.append(sid)
            outputs_list.append(outputs)
            targets_list.append(targets)
            t_ratio.append(tumor_ratio)
            t_patch_count.append(tumor_patch_count)

        all_outputs = torch.cat(outputs_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        tumor_ratios = torch.cat(t_ratio, dim=0)
        tumor_patch_counts = torch.cat(t_patch_count, dim=0)

        # Compute optimal threshold for binary in probability space
        if self.config['arch']['args']['out_channels'] > 1:
            threshold = None
        else:
            _, _, threshold = module_metric.Find_Optimal_Cutoff(
                all_outputs,   # logits
                all_targets,   # labels
                already_sigmoid=False  # let it sigmoid inside
            )

        # Build probs & preds lists
        if threshold is not None:
            probs_tensor = torch.sigmoid(all_outputs).flatten()
            probs = probs_tensor.cpu().tolist()
            preds = (probs_tensor >= threshold).int().cpu().tolist()
        else:
            # multiclass: take softmax + argmax
            sm = torch.softmax(all_outputs.squeeze(), dim=1)
            probs = sm.cpu().tolist()
            preds = torch.argmax(sm, dim=1).cpu().tolist()

        truths = all_targets.flatten().cpu().tolist()

        # Compute metrics exactly as in _valid_epoch but with our threshold
        auc_val  = module_metric.auc(all_outputs, all_targets, params=self.config)
        acc_val  = module_metric.accuracy(all_outputs, all_targets, threshold, params=self.config)
        prec_val = module_metric.precision(all_outputs, all_targets, threshold, params=self.config)
        sens_val = module_metric.sensitivity(all_outputs, all_targets, threshold, params=self.config)
        spec_val = module_metric.specificity(all_outputs, all_targets, threshold, params=self.config)
        f1_val   = module_metric.f1_score(all_outputs, all_targets, threshold, params=self.config)

        # Optional CSV / logging
        if verbose:
            import pandas as pd
            df = pd.DataFrame({
                'patient_id': patient_id,
                'slide_id':   slide_id,
                'prob':       probs,
                'pred':       preds,
                'truth':      truths,
                'threshold':  threshold
            })
            df.to_csv(self.log_dir / f'output_{mode}_fold{self.cv}.csv', index=False)
            self.logger.info(
                f"{mode.title()} |Fold: {self.cv}| AUC: {auc_val:.4f} | Acc: {acc_val:.4f} | "
                f"Prec: {prec_val:.4f} | Sens: {sens_val:.4f} | Spec: {spec_val:.4f} | F1: {f1_val:.4f}"
            )

        # Return 11 items to match unpack in train_cv.py
        return (
            auc_val, prec_val, sens_val, spec_val,
            acc_val, f1_val,
            patient_id, threshold,
            probs, preds, truths
        )

    def _forward_and_loss(self,
        model: torch.nn.Module,
        model_name: str,
        data: torch.Tensor,
        coords: torch.Tensor,
        targets: torch.Tensor,
        features_tme: Optional[torch.Tensor],
        criterion: Any,
        config: Dict[str, Any],
        supcon_criterion: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass and loss computation for different MIL models.
        Returns:
        outputs: raw logits
        loss: scalar tensor
        """
        if model_name == "CLAM":
            outputs, Y_prob, Y_hat, _, inst = model(data, label=targets)
            bag_loss = inst['instance_loss']
            loss = 0.5 * bag_loss + 0.5 * inst['instance_loss']
        elif model_name == "BasicMIL":
            outputs, Y_prob, Y_hat, _, _ = model(data)
            loss = criterion(Y_prob, targets, params=config)
        elif model_name == "TransMIL":
            res = model(data)
            outputs = res['logits']
            loss = criterion(outputs, targets, params=config)
        elif model_name == "GigaPath":
            outputs = model(data, coords)
            loss = criterion(outputs, targets, params=config)
        elif model_name == "DSMIL":
            ins_preds, bag_preds, attn = model(data)
            max_preds, _ = torch.max(ins_preds, 0, keepdim=True)
            bag_loss = (
                0.5 * criterion(max_preds, targets, params=config)
                + 0.5 * criterion(bag_preds, targets, params=config)
            )
            attn_probs = F.softmax(attn, dim=-1)
            div_loss = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
            loss = bag_loss + 0.1 * div_loss
            outputs = bag_preds
        elif model_name=="ACMIL":
            sub_preds, outputs, attn, slide_embeds = model(
                data, coords=coords, slide_meta=features_tme
            )
            sub_preds = sub_preds.squeeze()
            attn = attn.squeeze()
            if model.conf.n_token > 1:
                loss0 = criterion(
                    sub_preds,
                    targets.repeat_interleave(model.conf.n_token),
                    params=config
                )
            else:
                loss0 = torch.tensor(0.0, device=targets.device)
            loss1 = criterion(outputs, targets, params=config)
            supcon = supcon_criterion(slide_embeds.float(), targets)
            loss = loss0 + loss1 + 0.5 * supcon
        else:
            outputs = model(data)
            loss = criterion(outputs, targets, params=config)
        return outputs, loss
    
    def _safe_torch_load(self, path: Path, map_location):
        """
        Load a torch checkpoint robustly:
        - wait for file to exist and stabilize in size across two checks
        - retry once if the zip central directory error occurs
        """
        path = str(path)
        last_sz = -1
        # up to ~2s wait for stability
        for _ in range(4):
            if not os.path.exists(path):
                time.sleep(0.5); continue
            sz = os.path.getsize(path)
            if sz > 0 and sz == last_sz:
                break
            last_sz = sz
            time.sleep(0.5)

        try:
            return torch.load(path, weights_only=False, map_location=map_location)
        except RuntimeError as e:
            if "central directory" in str(e):
                # brief backoff + one retry
                time.sleep(1.0)
                return torch.load(path, weights_only=False, map_location=map_location)
            raise