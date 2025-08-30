import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler


def load_custom_dataset(data_root, batch_size, num_workers):
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((576, 768)),
            # transforms.CenterCrop(size=(50, 50)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((576, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load your custom training and validation datasets with ImbalancedDatasetSampler
    train_dataset = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=data_transforms['val'])

    # Create data loaders with ImbalancedDatasetSampler
    train_loader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=batch_size,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
