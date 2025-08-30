import os
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from loss import categorical_loss
from train_iter import train_iteration
from load_data import load_custom_dataset
import matplotlib.pyplot as plt

# Define the main function
def main():
    # Define source and destination paths
    source_path = "C:/THY_Database/Main_ADU/AIRE410/MVI/data"
    dest_path = "C:/THY_Database/Main_ADU/AIRE410/MVI/results"
    print("Chosen the paths")

    # Hyperparameters
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.001
    weight_decay = 1e-4  # L2 Regularization factor
    dropout_rate = 0.5   # Dropout rate
    num_classes = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print whether the model is running on GPU or CPU
    if device == 'cuda':
        print('Training on GPU')
    else:
        print('Training on CPU')

    # Load your custom dataset using load_data.py
    train_loader, val_loader = load_custom_dataset(data_root=source_path, batch_size=batch_size, num_workers=4)
    print("custom dataset loaded")

    # Initialize the model, add dropout layers, and move it to the device
    model = models.efficientnet_b2(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),  # Add dropout before the final layer
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)  # Move model to GPU or CPU
    print("model initiated with dropout layer")

    # Loss and optimizer with L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    print("Starting the loop")
    train_losses, train_accuracies, train_recalls, train_precisions, train_ious, train_f1_scores = train_iteration(
        model, train_loader, optimizer, categorical_loss, device, num_epochs
    )

    print('Training complete.')

    # Save the trained model
    os.makedirs(dest_path, exist_ok=True)
    model_path = os.path.join(dest_path, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    print("Model saved")

    # Plot all metrics
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')

    plt.subplot(2, 3, 3)
    plt.plot(train_recalls)
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training Recall')

    plt.subplot(2, 3, 4)
    plt.plot(train_precisions)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training Precision')

    plt.subplot(2, 3, 5)
    plt.plot(train_ious)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training IoU')

    plt.subplot(2, 3, 6)
    plt.plot(train_f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training F1 Score')

    plot_path = os.path.join(dest_path, 'training_metrics_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)

    print(f'Model weights saved in {model_path}')
    print(f'Training plot saved in {plot_path}')


if __name__ == '__main__':
    main()
