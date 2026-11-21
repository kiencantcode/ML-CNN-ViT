import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


def grad_cam_for_cnn(model, target_layer, images, targets):
    activations, gradients = {}, {}

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    was_training = model.training
    model.eval()
    try:
        images = images.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = model(images)
            score = logits.gather(1, targets.view(-1, 1)).sum()
            model.zero_grad(set_to_none=True)
            score.backward()
    finally:
        handle.remove()
        model.train(was_training)

    acts = activations["value"].detach()
    grads = gradients["value"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

    flat = cam.flatten(1)
    low = flat.min(dim=1).values.view(-1, 1, 1)
    high = flat.max(dim=1).values.view(-1, 1, 1)
    cam = (cam - low) / (high - low).clamp_min(1e-8)
    return cam.cpu().numpy(), logits.detach().cpu()


def overlay_heatmap(image, heatmap, alpha=0.5, cmap="inferno"):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    colored = colormaps[cmap](np.clip(heatmap, 0, 1))[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)


# Attention rollout for timm VisionTransformer models (DeiT-S, SmallViT).
# Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers".

def fuse_heads(attn, head_fusion="mean"):
    # attn: (B, heads, N, N) -> (B, N, N)
    if head_fusion == "mean":
        return attn.mean(dim=1)
    if head_fusion == "max":
        return attn.amax(dim=1)
    if head_fusion == "min":
        return attn.amin(dim=1)
    raise ValueError(f"Unknown head_fusion: {head_fusion}")


def _attention_hook(store, head_fusion):
    def hook(module, args, kwargs, output):
        x = args[0] if args else kwargs["x"]
        batch, tokens, _ = x.shape
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        attn = ((q * module.scale) @ k.transpose(-2, -1)).float().softmax(dim=-1)
        store.append(fuse_heads(attn, head_fusion))
    return hook


def collect_vit_attentions(model, images, head_fusion="mean"):
    attentions, handles = [], []
    for block in model.blocks:
        handles.append(block.attn.register_forward_hook(
            _attention_hook(attentions, head_fusion), with_kwargs=True))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(images)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if len(attentions) != len(model.blocks):
        raise RuntimeError(f"captured {len(attentions)} attention maps for {len(model.blocks)} blocks")
    return attentions, logits


def _discard_lowest(attn, discard_ratio):
    batch = attn.shape[0]
    flat = attn.reshape(batch, -1)
    k = int(flat.shape[1] * discard_ratio)
    if k == 0:
        return attn
    threshold = flat.kthvalue(k, dim=1).values.view(batch, 1, 1)
    kept = torch.where(attn <= threshold, torch.zeros_like(attn), attn)
    kept[:, 0, 0] = attn[:, 0, 0]
    return kept


def rollout_from_attentions(attentions, discard_ratio=0.0, residual_weight=0.5):
    batch, tokens, _ = attentions[0].shape
    eye = torch.eye(tokens, device=attentions[0].device, dtype=attentions[0].dtype)
    eye = eye.expand(batch, tokens, tokens)
    rollout = eye.clone()
    for attn in attentions:
        if discard_ratio > 0:
            attn = _discard_lowest(attn, discard_ratio)
        attn = residual_weight * eye + (1.0 - residual_weight) * attn
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout
    return rollout


def _normalize_per_image(maps):
    flat = maps.flatten(1)
    low = flat.min(dim=1).values.view(-1, 1, 1)
    high = flat.max(dim=1).values.view(-1, 1, 1)
    return (maps - low) / (high - low).clamp_min(1e-8)


def attention_rollout_for_vit(model, images, head_fusion="mean", discard_ratio=0.0):
    attentions, logits = collect_vit_attentions(model, images, head_fusion)
    rollout = rollout_from_attentions(attentions, discard_ratio)

    num_prefix = getattr(model, "num_prefix_tokens", 1)
    grid_h, grid_w = model.patch_embed.grid_size
    cls_to_patches = rollout[:, 0, num_prefix:]
    maps = cls_to_patches.reshape(-1, 1, grid_h, grid_w)
    maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    maps = _normalize_per_image(maps)
    return maps.cpu().numpy(), logits.detach().cpu()
