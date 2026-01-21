from typing import Dict

import torch
from torch import nn


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_fn = nn.KLDivLoss(reduction="batchmean")

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = loss_fn(log_probs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            pred = torch.argmax(logits, dim=-1)
            true = torch.argmax(targets, dim=-1)
            correct += (pred == true).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)

    return {"loss": avg_loss, "accuracy": accuracy}