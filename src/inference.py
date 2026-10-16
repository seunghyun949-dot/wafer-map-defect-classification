import numpy as np
import torch


def run_inference(model, input_tensor: torch.Tensor, device: torch.device):
    x = input_tensor.to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probabilities.argmax().item())
    return pred_idx, probabilities.cpu().numpy()
