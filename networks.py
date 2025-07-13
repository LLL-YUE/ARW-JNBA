"""
Loads model. 
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch

import dino.vision_transformer as vits

def get_model(arch, patch_size, device):

    # Initialize model with pretraining
    url = None
    if "vit" in arch:
        if arch == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    else:
        raise NotImplementedError

    for p in model.parameters():
        p.requires_grad = False
    if url is not None:
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )
    else:
        state_dict = torch.load("pretrained model/checkpoint0140.pth")['teacher']
        del state_dict['head.mlp.0.weight']
        del state_dict['head.mlp.0.bias']
        del state_dict['head.mlp.2.weight']
        del state_dict['head.mlp.2.bias']
        del state_dict['head.mlp.4.weight']
        del state_dict['head.mlp.4.bias']
        del state_dict['head.last_layer.weight_g']
        del state_dict['head.last_layer.weight_v']
        # state_dict = torch.load("pretrained model/dino_deitsmall16_pretrain.pth")
        msg = model.load_state_dict(state_dict, strict=True)
        print(
            "There is no reference weights available for this model => We use random weights."
        )
    model.eval()
    model.to(device)
    return model
