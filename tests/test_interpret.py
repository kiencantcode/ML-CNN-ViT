import torch
from timm.models.vision_transformer import VisionTransformer

from src.interpret import (attention_rollout_for_vit, collect_vit_attentions,
                           rollout_from_attentions)


def _tiny_vit():
    torch.manual_seed(0)
    return VisionTransformer(img_size=32, patch_size=8, embed_dim=32, depth=3,
                             num_heads=4, num_classes=10).eval()


def test_hooked_attention_matches_timm_softmax():
    # Our recomputed attention must equal what timm itself computes on the
    # non-fused path (captured at the input of attn_drop).
    model = _tiny_vit()
    images = torch.randn(2, 3, 32, 32)
    ours, _ = collect_vit_attentions(model, images, head_fusion="mean")

    reference, handles = [], []
    for block in model.blocks:
        block.attn.fused_attn = False
        handles.append(block.attn.attn_drop.register_forward_hook(
            lambda m, inp, out: reference.append(inp[0].mean(dim=1))))
    with torch.no_grad():
        model(images)
    for handle in handles:
        handle.remove()

    assert len(ours) == len(reference) == 3
    for a, b in zip(ours, reference):
        assert torch.allclose(a, b, atol=1e-5)


def test_rollout_is_row_stochastic():
    attentions = [torch.rand(2, 17, 17).softmax(dim=-1) for _ in range(4)]
    for ratio in (0.0, 0.9):
        rollout = rollout_from_attentions(attentions, discard_ratio=ratio)
        assert torch.allclose(rollout.sum(dim=-1), torch.ones(2, 17), atol=1e-5)


def test_identity_attention_gives_identity_rollout():
    eye = torch.eye(5).expand(1, 5, 5)
    rollout = rollout_from_attentions([eye, eye, eye])
    assert torch.allclose(rollout, eye)


def test_rollout_map_shape_and_range():
    model = _tiny_vit()
    maps, logits = attention_rollout_for_vit(model, torch.randn(3, 3, 32, 32))
    assert maps.shape == (3, 32, 32)
    assert logits.shape == (3, 10)
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6
