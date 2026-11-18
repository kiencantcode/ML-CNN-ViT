import timm
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from src.data import cifar_data_config


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes, width=64, dropout=0.1):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        self.stage1 = nn.Sequential(conv_bn_relu(3, c1), conv_bn_relu(c1, c1), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(conv_bn_relu(c1, c2), conv_bn_relu(c2, c2), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(conv_bn_relu(c2, c3), conv_bn_relu(c3, c3))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, num_classes)
        self.gradcam_layer = self.stage3[-1]

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))

    def get_classifier(self):
        return self.fc


def build_small_vit(model_cfg, num_classes):
    # Small ViT for 32x32 CIFAR, trained from scratch.
    # Default size (~1.2M params) matches SmallCNN(width=64) (~1.15M).
    return VisionTransformer(
        img_size=model_cfg.get("img_size", 32),
        patch_size=model_cfg.get("patch_size", 4),
        embed_dim=model_cfg.get("embed_dim", 128),
        depth=model_cfg.get("depth", 6),
        num_heads=model_cfg.get("num_heads", 4),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        drop_rate=model_cfg.get("dropout", 0.1),
        num_classes=num_classes,
    )

def build_model(model_cfg, num_classes):
    name = model_cfg["name"]
    if name == "small_cnn":
        model = SmallCNN(num_classes,
                         width=model_cfg.get("width", 64),
                         dropout=model_cfg.get("dropout", 0.1))
    elif name == "small_vit":
        model = build_small_vit(model_cfg, num_classes)
    else:
        model = timm.create_model(name, pretrained=model_cfg.get("pretrained", True),
                                  num_classes=num_classes)

    if model_cfg.get("freeze_backbone", False):
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    return model


def get_model_data_config(model, model_cfg, data_cfg):
    if model_cfg["name"] in ("small_cnn", "small_vit"):
        return cifar_data_config(data_cfg.get("image_size", 32))
    return timm.data.resolve_model_data_config(model)


def get_gradcam_layer(model, model_cfg):
    name = model_cfg["name"]
    if name == "small_cnn":
        return model.gradcam_layer
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("convnext"):
        return model.stages[-1]
    raise ValueError(f"No Grad-CAM layer defined for {name}")


def get_param_groups(model, lr_head, lr_backbone, weight_decay):
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = {
        ("head", True): [], ("head", False): [],
        ("backbone", True): [], ("backbone", False): [],
    }
    for param in model.parameters():
        if not param.requires_grad:
            continue
        part = "head" if id(param) in head_ids else "backbone"
        use_decay = param.ndim > 1
        groups[(part, use_decay)].append(param)

    param_groups = []
    for (part, use_decay), params in groups.items():
        if not params:
            continue
        param_groups.append({
            "name": f"{part}_{'decay' if use_decay else 'no_decay'}",
            "params": params,
            "lr": lr_head if part == "head" else lr_backbone,
            "weight_decay": weight_decay if use_decay else 0.0,
        })
    return param_groups
