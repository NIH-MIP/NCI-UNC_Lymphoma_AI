import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
import pandas as pd
import os
import re
import torch
import numpy as np
import random
from scipy import interpolate
from torch import inf
import torch.distributed as dist
import pickle
import h5py
import tempfile
from importlib import import_module
from itertools import repeat
from functools import partial, update_wrapper


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def to_one_hot(x, C=1, tensor_class=torch.cuda.FloatTensor):
    """ One-hot a batched tensor of shape (B, ...) into (B, C, ...) """
    x_one_hot = tensor_class(x.size(0), C, *x.shape[1:]).zero_()
    x_one_hot = x_one_hot.scatter_(1, x.unsqueeze(1), 1)
    return x_one_hot


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optim's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _init_fn(worker_id, SEED):
    seed_torch(SEED)


def auto_resume_helper(output_dir, fold=None):
    checkpoints = os.listdir(output_dir)
    if fold is not None:
        checkpoints = [ckpt for ckpt in checkpoints if (ckpt.endswith('pth') and ckpt.startswith(str(fold),4,5))]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints if d.endswith('pth')], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    is_dist_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    rt = tensor.clone() if is_dist_initialized and isinstance(tensor, torch.Tensor) else tensor
    if is_dist_initialized:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
    return rt


def save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pkl(filename):
    loader = open(filename,'rb')
    file = pickle.load(loader)
    loader.close()
    return file

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def save_model(conf, epoch, model, optimizer, is_best=False, is_last=False):
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': conf,
    }

    checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-%s.pth' % epoch)

    # record the checkpoint with best validation accuracy
    if is_best:
        checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-best.pth')

    if is_last:
        checkpoint_path = os.path.join(conf.ckpt_dir, 'checkpoint-last.pth')

    torch.save(to_save, checkpoint_path)


# Create a temporary directory for caching
TMP_DIR = tempfile.mkdtemp()

def get_tmp_path(bag_id):
    """Generate a file path for a specific bag ID."""
    return os.path.join(TMP_DIR, f"bag_{bag_id}.pkl")


def cleanup_cache():
    """Delete all cached files after training."""
    global TMP_DIR
    if os.path.exists(TMP_DIR):
        for file in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, file))
        os.rmdir(TMP_DIR)
        print("Cache cleaned up.")