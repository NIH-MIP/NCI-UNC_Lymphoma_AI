import model.arch.CLAM as clam_base
import model.arch.dsmil as ds_mil
import model.arch.basic_MIL as bmil_base
import model.arch.attention_MIL as abmil_base
from model.arch.TransMIL import TransMIL
import model.arch.gigapath as gigapth
import model.arch.ACMIL as acmil

import torch
from torch import nn





############### MIL models ########################

def transmil(**kwargs):
    model = TransMIL(n_classes=kwargs["out_channels"],embed_dim=kwargs["embed_dim"])
    model.name = "TransMIL"
    return model

def clam(model_type = "sb", **kwargs):
    if model_type == "sb":
        model = clam_base.CLAM_SB(n_classes=kwargs["out_channels"],embed_dim=kwargs["embed_dim"],dropout=0.25)
    elif model_type == 'mb':
        model = clam_base.CLAM_MB(n_classes=kwargs["out_channels"],embed_dim=kwargs["embed_dim"],dropout=0.25)
    model.name = "CLAM"
    return model


def abmil(**kwargs):
    model = abmil_base.DeepMIL(num_cls=kwargs["out_channels"],dim_in=kwargs["embed_dim"], dim_hid=kwargs["D_inner"], drop_rate=0.0,
                               pooling='gated_attention')
    model.name = "Attention_MIL"
    return model

def basic_mil(**kwargs):
    if kwargs["out_channels"] <=2:
        model = bmil_base.MIL_fc(n_classes=kwargs["out_channels"],embed_dim=kwargs["embed_dim"])
    elif kwargs["out_channels"] > 2:
        model = bmil_base.MIL_fc_mc(n_classes=kwargs["out_channels"],embed_dim=kwargs["embed_dim"])
    model.name = "BasicMIL"
    return model

def gigapath(**kwargs):
    class ClassificationHead(nn.Module):
        def __init__(
                self, input_dim, latent_dim, feat_layer, n_classes=2, model_arch="gigapath_slide_enc12l768d",
                pretrained="hf_hub:prov-gigapath/prov-gigapath", freeze=False,**kwargs):
            super(ClassificationHead, self).__init__()

            # setup the slide encoder
            self.feat_layer = [eval(x) for x in feat_layer.split("-")]
            self.feat_dim = len(self.feat_layer) * latent_dim
            self.slide_encoder = gigapth.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

            # whether to freeze the pretrained model
            if freeze:
                print("Freezing Pretrained GigaPath model")
                for name, param in self.slide_encoder.named_parameters():
                    param.requires_grad = False
                print("Done")
            # set up the classifier
            self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])

        def forward(self, images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            """
            Arguments:
            ----------
            images: torch.Tensor
                The input images with shape [N, L, D]
            coords: torch.Tensor
                The input coordinates with shape [N, L, 2]
            """
            # inputs: [N, L, D]
            if len(images.shape) == 2:
                images = images.unsqueeze(0)
            assert len(images.shape) == 3
            # forward GigaPath slide encoder
            img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True)
            img_enc = [img_enc[i] for i in self.feat_layer]
            img_enc = torch.cat(img_enc, dim=-1)
            # classifier
            h = img_enc.reshape([-1, img_enc.size(-1)])
            logits = self.classifier(h)
            return logits

    model = ClassificationHead(input_dim=1536,latent_dim=768,feat_layer='11', n_classes=kwargs["out_channels"])
    model.name = "GigaPath"
    return model


def dsmil(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.D_feat = kwargs["embed_dim"]
    conf.n_class = kwargs["out_channels"]
    i_classifier = ds_mil.FCLayer(in_size = conf.D_feat, out_size=conf.n_class)
    b_classifier = ds_mil.BClassifier(conf, nonlinear=False)
    model = ds_mil.MILNet(i_classifier, b_classifier)
    model.name = "DSMIL"
    model.conf = conf
    return model


def acmil_ga(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.mask_drop = kwargs["mask_drop"]
    conf.n_masked_patch = kwargs["n_masked_patch"]
    conf.n_class = kwargs["out_channels"]
    conf.D_feat = kwargs["embed_dim"]
    model = acmil.ACMIL_GA(conf)
    # model = acmil_original.ACMIL_GA(conf, D = 128)
    model.name = "ACMIL"
    model.conf = conf

    if kwargs["pretrained_model_path"] is not None:
        path = kwargs['pretrained_model_path']

        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('_mil_encoder') and not k.startswith('_mil_encoder.fc') and "classifier" not in k:
                # remove prefix
                state_dict[k[len("_mil_encoder."):]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        print("pretrained model loaded successfully! <============>")
        print("pretrained model name: %s" % kwargs['pretrained_model_path'])
    return model



def acmil_mha(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.mask_drop = kwargs["mask_drop"]
    conf.n_masked_patch = kwargs["n_masked_patch"]
    conf.n_class = kwargs["out_channels"]
    conf.D_feat = kwargs["embed_dim"]
    model = acmil.ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop,
                            n_class=kwargs["out_channels"], D_feat=kwargs["embed_dim"])
    model.name = "ACMIL"
    model.conf = conf
    return model

