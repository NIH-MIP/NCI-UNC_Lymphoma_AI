from functools import partial
import torch.distributed as dist
import torch
from torch import optim as optim
from optim.custom_optimizer import lookahead, radam


def build_optimizer(config, model, filter_bias_and_bn=True):
    """
    Build optim, set weight decay of normalization to 0 by default.
    """
    weight_decay = config['optimizer']['args']['weight_decay']
    # === Compute scaled base LR ===
    base_lr = config['optimizer']['args']['lr']
    batch_size = config['data_loader_train']['args']['batch_size']
    accumulate_steps = config['trainer'].get('accumulate_steps', 1)

    opt_type = config['optimizer']['type'].lower()
    opt_split = opt_type.split('_')
    opt_lower = opt_split[-1]
    
    if config['distributed']:
        linear_scaled_lr = base_lr * batch_size * dist.get_world_size() / 96.0 # devide large value only when training large batch
    else:
        linear_scaled_lr = base_lr * batch_size
    if accumulate_steps>1:
        linear_scaled_lr = linear_scaled_lr * accumulate_steps
    
    encoder_lr = config['optimizer']['args'].get('encoder_lr', linear_scaled_lr)
    mil_lr = config['optimizer']['args'].get('mil_lr', linear_scaled_lr)

    # === Setup param groups ===
    if hasattr(model, 'feature_encoder') and hasattr(model, '_mil_encoder'):
        # Get skip list for both submodules if applicable
        skip = getattr(model, 'no_weight_decay', lambda: set())()
        encoder_params = add_weight_decay(model.feature_encoder, weight_decay, skip)
        mil_params = add_weight_decay(model._mil_encoder, weight_decay, skip)

        # Set learning rates for each group
        for group in encoder_params:
            group['lr'] = encoder_lr
        for group in mil_params:
            group['lr'] = mil_lr

        parameters = encoder_params + mil_params
        weight_decay = 0.  # Already handled per param group
    else:
        # Default handling for single module models
        if weight_decay and filter_bias_and_bn:
            skip = getattr(model, 'no_weight_decay', lambda: set())()
            parameters = add_weight_decay(model, weight_decay, skip)
            weight_decay = 0.
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config['optimizer']['args']['momentum'], nesterov=True,
                              lr=linear_scaled_lr, weight_decay=weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config['optimizer']['args']['eps'], betas=config['optimizer']['args']['betas'],
                                lr=linear_scaled_lr, weight_decay=weight_decay)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, eps=config['optimizer']['args']['eps'], betas=config['optimizer']['args']['betas'],
                                lr=linear_scaled_lr, weight_decay=weight_decay)
    elif opt_lower == 'radam':
        optimizer = optim.RAdam(parameters, eps=config['optimizer']['args']['eps'], betas=config['optimizer']['args']['betas'],
                                lr=linear_scaled_lr, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            for p in group['params']:
                st = optimizer.state[p]
                # these must match p.data’s device/dtype:
                st.setdefault('exp_avg',    torch.zeros_like(p.data))
                st.setdefault('exp_avg_sq', torch.zeros_like(p.data))
                # but step *must* be a CPU float32 singleton:
                st.setdefault('step',       torch.zeros(1, dtype=torch.float32))
    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = lookahead.Lookahead(optimizer)
    return optimizer


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())
