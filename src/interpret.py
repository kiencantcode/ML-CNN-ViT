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
