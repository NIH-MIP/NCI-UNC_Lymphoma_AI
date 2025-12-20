from model.loss_util import custom_loss
from sklearn.metrics import normalized_mutual_info_score as nmi_score
import monai
import timm.loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from typing import Optional, Dict, Any


multiclass = False


def cross_entropy_loss(output, target, params=None):
    # weights = [2, 1, 1]
    # class_weights = torch.FloatTensor(weights)
    criterion = torch.nn.CrossEntropyLoss(reduction=params['loss']['args']['reduction'])
    return criterion(output, target.long())


def contrastive_loss(output, target):
    output_1 = output[0]
    output_2 = output[1]
    criterion = custom_loss.ContrastiveLoss(margin=2.0)
    return criterion(output_1, output_2, target)


def mutualinfo_loss(output, target):
    output_1 = output[0]
    output_2 = output[1]
    criterion = torch.nn.BCEWithLogitsLoss()
    mutualinfo = nmi_score(output_1.cpu().numpy(), output_2.cpu().numpy())
    return criterion(torch.from_numpy(mutualinfo).float(), target)


def bce_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    params: Optional[Any] = None,
) -> torch.Tensor:
    """
    Binary focal loss with logits, supporting alpha/gamma and optional pos_weight.

    Args:
      logits: Tensor of shape [N] or [N,1]
      target: Tensor of shape [N] or [N,1], values 0 or 1
      params: config object or dict containing params['loss']['args'], e.g.
        {
          "alpha": 0.5,
          "gamma": 2.0,
          "reduction": "mean",
          "label_smoothing": 0.0,
          "pos_weight": 2.33      # optional, like BCEWithLogitsLoss
        }
    """
    # pull out the loss args dict
    loss_args   = params['loss']['args'] if params is not None else {}
    alpha       = loss_args.get('alpha', 1.0)
    gamma       = loss_args.get('gamma', 2.0)
    reduction   = loss_args.get('reduction', 'mean')
    smoothing   = loss_args.get('label_smoothing', 0.0)
    pos_weight  = loss_args.get('pos_weight', None)

    # align shapes: [N] -> [N,1]
    if logits.dim() > target.dim():
        target = target.unsqueeze(1)
    target = target.float()

    # label smoothing toward 0.5
    if smoothing > 0:
        target = target * (1 - smoothing) + 0.5 * smoothing

    # prepare pos_weight tensor if provided
    if pos_weight is not None:
        # can be scalar float or tensor-like; broadcast safely
        pos_w = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    else:
        pos_w = None

    # per-sample BCE with optional pos_weight
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction='none',
        pos_weight=pos_w,
    )

    # p_t = p if y=1 else (1-p)
    p = torch.sigmoid(logits)
    p_t = target * p + (1.0 - target) * (1.0 - p)

    # alpha-balancing factor (you can set alpha=0.5 if using pos_weight for imbalance)
    alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)

    # focal weighting
    focal_weight = alpha_factor * (1.0 - p_t).pow(gamma)

    loss = focal_weight * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def bce_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    params: Optional[Any] = None,
) -> torch.Tensor:
    """
    Standard BCE-with-logits loss with optional:
      - label smoothing
      - pos_weight support via params['loss']['pos_weight']
    """
    # ---- Extract config ---- #
    if params is not None:
        loss_args = params['loss']['args']
        pos_weight_val = loss_args.get('pos_weight', None)
        la_cfg = loss_args.get('logit_adjustment', None)
    else:
        loss_args = {}
        pos_weight_val = None
        la_cfg = None

    reduction = loss_args.get('reduction', 'mean')
    smoothing = loss_args.get('label_smoothing', 0.0)

    # ---- Shape normalization ---- #
    if logits.dim() > target.dim():
        target = target.unsqueeze(1)
    target = target.float()

    # ---- Label smoothing ---- #
    if smoothing > 0:
        # Smooth toward 0.5 for binary
        target = target * (1.0 - smoothing) + 0.5 * smoothing

    # ---- pos_weight handling ---- #
    if pos_weight_val is not None:
        # Accept float or tensor
        if isinstance(pos_weight_val, (float, int)):
            pos_weight = torch.tensor([float(pos_weight_val)], device=logits.device)
        elif isinstance(pos_weight_val, torch.Tensor):
            pos_weight = pos_weight_val.to(logits.device)
            if pos_weight.numel() != 1:
                raise ValueError("pos_weight must be a scalar for binary BCE.")
        else:
            raise TypeError("pos_weight must be float or torch.Tensor.")

    else:
        pos_weight = None

    if la_cfg is not None and la_cfg.get("enabled", False):
        # class prior: pi_pos
        pi_pos = torch.tensor(la_cfg["pi_pos"], device=logits.device, dtype=logits.dtype)
        pi_neg = 1.0 - pi_pos
        
        tau = la_cfg.get("tau", 1.0)

        # bias = τ * log(pi_pos / pi_neg)
        bias = tau * torch.log(pi_pos / pi_neg)

        # Add bias to logits **before** BCE-with-logits
        logits = logits + bias

    # ---- BCE-with-logits ---- #
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction=reduction,
        pos_weight=pos_weight,   # may be None (PyTorch handles it)
    )
