import torch
import torch.nn as nn


def categorical_loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss

def hinge_loss(outputs, targets):
    margin = 1.0  # Define the margin for the hinge loss

    # Calculate hinge loss
    loss = torch.mean(torch.max(0, margin - outputs[torch.arange(len(outputs)), targets] +
                       (outputs - outputs[torch.arange(len(outputs)), targets].unsqueeze(1)).clamp(min=0)))

    return loss