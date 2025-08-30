import torch

def train_iteration(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    losses = []
    accuracies = []
    recalls = []
    precisions = []
    ious = []
    f1_scores = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        tp, fp, fn = 0, 0, 0  # True Positives, False Positives, False Negatives

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Calculate TP, FP, FN for metrics
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = correct_predictions / total_samples

        # Calculate additional metrics
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Append the metrics for this epoch
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)
        recalls.append(recall)
        precisions.append(precision)
        ious.append(iou)
        f1_scores.append(f1_score)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
              f"Recall: {recall:.4f}, Precision: {precision:.4f}, IoU: {iou:.4f}, F1 Score: {f1_score:.4f}")

    return losses, accuracies, recalls, precisions, ious, f1_scores  # Return all metrics
