import numpy as np
import torch
import torch.nn.functional as F

from .config import GRADCAM_TARGET_LAYER_INDEX, INPUT_SIZE


def compute_gradcam(model, input_tensor, target_idx, device):
    x = input_tensor.to(device)
    x.requires_grad_(True)

    activations = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.retain_grad()

    target_layer = model.features[GRADCAM_TARGET_LAYER_INDEX]
    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, target_idx].backward()

        feature_map = activations["value"]
        gradients = feature_map.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * feature_map).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam
    finally:
        handle.remove()
