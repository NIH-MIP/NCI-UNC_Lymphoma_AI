import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .nystrom_attention import NystromAttention
from timm.models.layers import trunc_normal_

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            head_fusion='max'
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, embed_dim):
        super(TransMIL, self).__init__()
        self.dim_size = 512
        self.pool = nn.AdaptiveMaxPool2d((20480,embed_dim))
        self.pos_layer = PPEG(dim=self.dim_size)
        self._fc1 = nn.Sequential(nn.Linear(embed_dim, self.dim_size), nn.ReLU())
        # self._fc2 = nn.Sequential(nn.Linear(512, self.dim_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=self.dim_size)
        self.layer2 = TransLayer(dim=self.dim_size)
        self.norm = nn.LayerNorm(self.dim_size)
        self._fc3 = nn.Linear(self.dim_size, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        assert x.shape[0] == 1 # [1, n, 1024], single bag
        ft = x.float()  # [B, n, 1024]
        ft = self.pool(ft)
        ft = self._fc1(ft)  # [B, n, 512]
        # ft = self._fc2(ft)

        # rand_idx = torch.randperm(ft.size(2),device=x.device)
        # idx = rand_idx[:512]
        # ft = ft.index_select(2,idx)

        # ---->pad
        H = ft.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        ft = torch.cat([ft, ft[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = ft.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, ft), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc3(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2,embed_dim=1024).cuda()
    print(model.eval())
    results_dict = model(data=data)
    print(results_dict)
