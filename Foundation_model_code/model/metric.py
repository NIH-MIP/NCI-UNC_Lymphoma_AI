from torchmetrics.functional.segmentation import dice_score as tm_dice, mean_iou as tm_iou, hausdorff_distance as tm_hd
from torchmetrics.classification import BinaryROC, MultilabelROC
import sklearn.metrics as skm
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    binary_accuracy,
    multiclass_accuracy,
    binary_precision,
    multiclass_precision,
    binary_recall,
    multiclass_recall,
    binary_specificity,
    multiclass_specificity,
    binary_f1_score,
    multiclass_f1_score,
    binary_auroc,
)
from typing import Optional, Tuple


def _is_multiclass(output: torch.Tensor) -> bool:
    return output.ndim == 2 and output.shape[1] > 1

def _ensure_flat(output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten and move to correct dtype."""
    out = output.view(-1, output.shape[-1]) if _is_multiclass(output) else output.view(-1)
    tgt = target.view(-1).long()
    return out, tgt



@torch.no_grad()
def roc_curve(output: torch.Tensor, target: torch.Tensor):
    """
    Returns full (fpr, tpr, thresholds) just like sklearn.metrics.roc_curve,
    but computed on the GPU via torchmetrics.

    - For binary: returns three 1D torch.Tensor.
    - For multilabel: returns lists of tensors, one per label.
    """
    # binary if logits shape is [N] or [N,1]
    if output.ndim == 1 or (output.ndim == 2 and output.size(1) == 1):
        probs = torch.sigmoid(output).flatten()
        truth = target.flatten().int()
        roc = BinaryROC(thresholds=None).to(probs.device)
        fpr, tpr, thr = roc(probs, truth)
        return fpr, tpr, thr

    # otherwise multilabel: logits shape [N, C]
    C = output.size(1)
    probs = torch.sigmoid(output)
    truth = target
    if truth.ndim == 1:
        # class indices → one-hot
        truth = torch.nn.functional.one_hot(truth, num_classes=C)
    truth = truth.int()
    roc = MultilabelROC(num_labels=C, thresholds=None).to(probs.device)
    fpr_list, tpr_list, thr_list = roc(probs, truth)

    # return as lists for compatibility
    return list(fpr_list), list(tpr_list), list(thr_list)


@torch.no_grad()
def Find_Optimal_Cutoff(output: torch.Tensor,
                         target: torch.Tensor,
                         already_sigmoid: bool = False):
    """
    Returns (best_fpr, best_tpr, best_threshold) by maximizing (tpr - fpr).

    IMPORTANT: For binary classification, this assumes `output` is P(y==1) in [0,1]
    unless `already_sigmoid=True` is passed. This makes the returned threshold
    a *probability threshold* that you can compare against sigmoid(logit).
    """
    # Flatten to 1D
    output = output.detach().float().view(-1)
    target = target.detach().float().view(-1)

    # Ensure probability space for binary case
    if not already_sigmoid:
        output = torch.sigmoid(output)

    fpr, tpr, thr = roc_curve(output, target)  # your wrapper around sklearn

    # binary case
    if isinstance(thr, torch.Tensor):
        j = tpr - fpr
        idx = torch.argmax(j)
        return float(fpr[idx]), float(tpr[idx]), float(thr[idx])

    # multilabel case (unchanged)
    best_fpr, best_tpr, best_thr = [], [], []
    for fi, ti, thi in zip(fpr, tpr, thr):
        j = ti - fi
        idx = torch.argmax(j)
        best_fpr.append(float(fi[idx]))
        best_tpr.append(float(ti[idx]))
        best_thr.append(float(thi[idx]))
    return best_fpr, best_tpr, best_thr

    

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor,
             threshold: float = 0.5, params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        # apply softmax → [N, C] probabilities
        probs = F.softmax(out, dim=1)
        return multiclass_accuracy(probs, tgt,
                                   num_classes=probs.shape[1],
                                   average="micro").item()
    else:
        # binary
        probs = torch.sigmoid(out)
        return binary_accuracy(probs, tgt, threshold=threshold).item()


@torch.no_grad()
def precision(output: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        probs = F.softmax(out, dim=1)
        return multiclass_precision(probs, tgt,
                                    num_classes=probs.shape[1],
                                    average="macro").item()
    else:
        probs = torch.sigmoid(out)
        return binary_precision(probs, tgt, threshold=threshold).item()


@torch.no_grad()
def sensitivity(output: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5, params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        probs = F.softmax(out, dim=1)
        return multiclass_recall(probs, tgt,
                                 num_classes=probs.shape[1],
                                 average="macro").item()
    else:
        probs = torch.sigmoid(out)
        return binary_recall(probs, tgt, threshold=threshold).item()

@torch.no_grad()
def specificity(output: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5, params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        probs = F.softmax(out, dim=1)
        return multiclass_specificity(probs, tgt,
                                      num_classes=probs.shape[1],
                                      average="macro").item()
    else:
        probs = torch.sigmoid(out)
        return binary_specificity(probs, tgt, threshold=threshold).item()

@torch.no_grad()
def f1_score(output: torch.Tensor, target: torch.Tensor,
             threshold: float = 0.5, params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        probs = F.softmax(out, dim=1)
        return multiclass_f1_score(probs, tgt,
                                   num_classes=probs.shape[1],
                                   average="macro").item()
    else:
        probs = torch.sigmoid(out)
        return binary_f1_score(probs, tgt, threshold=threshold).item()

@torch.no_grad()
def auc(output: torch.Tensor, target: torch.Tensor,
        params: Optional[dict] = None) -> float:
    out, tgt = _ensure_flat(output, target)
    if _is_multiclass(output):
        # multiclass OVR via sklearn
        probs = F.softmax(out.float(), dim=1).cpu().numpy()
        labels = tgt.cpu().numpy()
        return float(skm.roc_auc_score(labels, probs, multi_class="ovr"))
    else:
        # binary
        probs = torch.sigmoid(out)
        return binary_auroc(probs, tgt).item()