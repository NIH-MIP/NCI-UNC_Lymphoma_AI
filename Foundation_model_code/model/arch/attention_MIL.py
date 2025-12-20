import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DeepMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task.

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """

    def __init__(self, dim_in=1024, dim_hid=512, num_cls=1, use_feat_proj=True, drop_rate=0.5,
                 pooling='attention', pred_head='default'):
        super(DeepMIL, self).__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']

        if use_feat_proj:
            self.feat_proj = Feat_Projecter(dim_in, dim_in)
        else:
            self.feat_proj = None

        if pooling == 'gated_attention':
            self.sigma = Gated_Attention_Pooling(dim_in, dim_hid, dropout=drop_rate)
        elif pooling == 'attention':
            self.sigma = Attention_Pooling(dim_in, dim_hid)
        else:
            self.sigma = pooling

        if pred_head == 'default':
            self.g = nn.Linear(dim_in, num_cls)
        else:
            self.g = nn.Sequential(
                nn.Linear(dim_in, dim_hid),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate),
                nn.Linear(dim_hid, num_cls)
            )

    def forward(self, X, ret_with_attn=False):
        """
        X: initial bag features, with shape of [b, K, d]
           where b = 1 for batch size, K is the instance size of this bag, and d is feature dimension.
        """
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)

        # global pooling function sigma
        # [B, N, C] -> [B, C]
        if self.sigma == 'mean':
            X_vec = torch.mean(X, dim=1)
        elif self.sigma == 'max':
            X_vec, _ = torch.max(X, dim=1)
        else:
            X_vec, raw_attn = self.sigma(X)

        # prediction head w/o norm
        # [B, C] -> [B, num_cls]
        logit = self.g(X_vec)

        if ret_with_attn:
            attn = raw_attn.detach()
            return logit, attn  # [B, num_cls], [B, N]
        else:
            return logit


class Gated_Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim, hid_dim, dropout=0.5):
        super(Gated_Attention_Pooling, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x, ret_raw_attn=False):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        emb = self.fc1(x) # [B, N, d']
        scr = self.score(x) # [B, N, d'] \in [0, 1]
        new_emb = emb.mul(scr)
        A_ = self.fc2(new_emb) # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1) # [B, 1, N]
        A = F.softmax(A_, dim=2) # [B, 1, N]
        out = torch.matmul(A, x).squeeze(1) # [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1) # [B, N]
            return out, A_
        else:
            A = A.squeeze(1) # [B, N]
            return out, A


class Attention_Pooling(nn.Module):
    """Global Attention Pooling implemented by
    [Ilse et al. Attention-based Deep Multiple Instance Learning. ICML 2018.]
    """
    def __init__(self, in_dim=1024, hid_dim=512):
        super(Attention_Pooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x, ret_raw_attn=True):
        """
        x -> out : [B, N, d] -> [B, d]
        """
        A_ = self.attention(x)  # [B, N, 1]
        A_ = torch.transpose(A_, 2, 1)  # [B, 1, N]
        attn = F.softmax(A_, dim=2)  # [B, 1, N]
        out = torch.matmul(attn, x).squeeze(1)  # [B, 1, N] bmm [B, N, d] = [B, 1, d]
        if ret_raw_attn:
            A_ = A_.squeeze(1)
            return out, A_
        else:
            A = A.squeeze(1)
            return out, A

class Feat_Projecter(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024):
        super(Feat_Projecter, self).__init__()
        self.projecter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            B, N, C = x.shape
            x = self.projecter(x.view(B * N, C)).view(B, N, -1)
        else:
            x = self.projecter(x)
        return x