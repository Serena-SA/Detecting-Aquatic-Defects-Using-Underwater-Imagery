import torch

# Load the existing model state_dict
old_model_path = "model_weights.pth" # Here is actually the whole path.
new_model_path = "entire_model_direct.pth" # Also here needs a path.

# Reinitialize the model architecture
from torchvision import models
import torch.nn as nn

# Match the architecture used during training
model = models.efficientnet_b2(weights="IMAGENET1K_V1")
num_classes = 3  # or the number you used
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

# Load the state_dict into the model
model.load_state_dict(torch.load(old_model_path))

# Save the entire model (architecture + weights)
torch.save(model, new_model_path)
print("Model re-saved successfully at", new_model_path)
