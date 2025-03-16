import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# -------------------------------
# Utility Functions
# -------------------------------


def compute_texture_mask(image_np):
    """
    Compute an adaptive frequency mask using Canny edge detection.
    image_np: numpy image in BGR format (uint8, shape H x W x C)
    Returns a float32 mask in [0,1] with same spatial dims.
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    # Normalize and smooth the mask
    mask = edges.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    # Expand dims to match channel count
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, image_np.shape[2], axis=-1)
    return mask


def simulate_compression(image_np, quality=75):
    """
    Simulate JPEG compression on an image.
    image_np: numpy image in uint8 format.
    Returns the compressed image (uint8).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", image_np, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def dct2_channel(channel):
    """
    Apply 2D DCT to a single channel (float32).
    """
    return cv2.dct(channel)


def idct2_channel(channel_dct):
    """
    Apply inverse 2D DCT to a single channel.
    """
    return cv2.idct(channel_dct)


# -------------------------------
# Adaptive Frequency Attack
# -------------------------------


def adaptive_frequency_attack(
    image_tensor, model, target_label, criterion, steps=10, alpha=0.01, device="cuda"
):
    """
    Generate a perturbed image (unlearnable example) using an adaptive frequency attack.
    image_tensor: shape [1, C, H, W], float [0,1].
    model: surrogate model used to compute loss. Should be in eval mode.
    target_label: ground-truth label (int) for computing loss.
    criterion: loss function (e.g., cross entropy).
    """
    perturbed = image_tensor.clone().to(device)
    perturbed.requires_grad = True

    for _ in range(steps):
        outputs = model(perturbed)
        # Use negative cross-entropy to *maximize* loss (make the example unlearnable)
        loss = -criterion(outputs, torch.tensor([target_label], device=device))
        loss.backward()
        grad = perturbed.grad.detach()  # [1, C, H, W]

        # Convert tensor to numpy (H x W x C, in [0,255] uint8) for DCT processing
        image_np = perturbed.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        grad_np = grad.cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Process each channel using DCT
        dct_channels = []
        grad_dct_channels = []
        for c in range(image_np.shape[2]):
            channel = image_np[:, :, c].astype(np.float32)
            dct_channel = dct2_channel(channel)
            dct_channels.append(dct_channel)

            grad_channel = grad_np[:, :, c]
            grad_dct = dct2_channel(grad_channel)
            grad_dct_channels.append(grad_dct)
        dct_array = np.stack(dct_channels, axis=2)
        grad_dct_array = np.stack(grad_dct_channels, axis=2)

        # Compute adaptive mask from original image texture
        mask = compute_texture_mask(image_np)

        # Apply mask to gradient in DCT domain
        grad_dct_masked = grad_dct_array * mask

        # Update DCT coefficients with a PGD step
        updated_dct = dct_array + alpha * np.sign(grad_dct_masked)

        # Inverse DCT on each channel
        updated_channels = []
        for c in range(updated_dct.shape[2]):
            idct_channel = idct2_channel(updated_dct[:, :, c])
            updated_channels.append(idct_channel)
        updated_image = np.stack(updated_channels, axis=2)

        # Clip and normalize to [0,1]
        updated_image = np.clip(updated_image, 0, 255) / 255.0

        # Convert back to tensor [1, C, H, W]
        updated_tensor = (
            torch.tensor(updated_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
        )

        # Simulate JPEG compression
        updated_np = updated_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        updated_np_uint8 = np.clip(updated_np * 255, 0, 255).astype(np.uint8)
        compressed_np = simulate_compression(updated_np_uint8, quality=75)
        compressed_tensor = (
            torch.tensor(compressed_np).permute(2, 0, 1).unsqueeze(0).to(device).float()
            / 255.0
        )

        # Prepare for next iteration
        perturbed = compressed_tensor.clone().detach()
        perturbed.requires_grad = True

    return perturbed.detach()


# -------------------------------
# Dataset Generation: Unlearnable Examples
# -------------------------------


class UnlearnableCIFAR10(torch.utils.data.Dataset):
    """
    Wrap CIFAR10 to generate unlearnable examples on the fly.
    """

    def __init__(
        self, base_dataset, model, criterion, steps=10, alpha=0.01, device="cuda"
    ):
        self.base_dataset = base_dataset
        self.model = model
        self.criterion = criterion
        self.steps = steps
        self.alpha = alpha
        self.device = device

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = image.unsqueeze(0).to(self.device)  # shape [1, C, H, W]
        # Generate the perturbed (unlearnable) image
        perturbed = adaptive_frequency_attack(
            image,
            self.model,
            label,
            self.criterion,
            steps=self.steps,
            alpha=self.alpha,
            device=self.device,
        )
        # Move back to CPU and remove batch dimension
        perturbed = perturbed.squeeze(0).cpu()
        return perturbed, label


# -------------------------------
# Testing: Train a Classifier
# -------------------------------


def train_classifier(train_loader, test_loader, num_epochs=10, device="cuda"):
    """
    Train a classifier (ResNet18) on the given train_loader and evaluate on test_loader.
    """
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}")

        # Evaluation on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total * 100.0
        print(f"Test Accuracy: {accuracy:.2f}%")
    return model


def test_classifier(model, test_loader, device="cuda"):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    avg_loss = 0.0

    with torch.no_grad():  # No need to track gradients during inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            avg_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss /= len(test_loader)

    print(f"Test Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f}")
    return accuracy, avg_loss


# -------------------------------
# Main: Generate Datasets and Compare
# -------------------------------


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    # Define transforms for CIFAR10 (using ToTensor converts to [0,1])
    transform = transforms.Compose([transforms.ToTensor()])

    # Load CIFAR10 train and test datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Load a surrogate model for generating unlearnable examples.
    # For demonstration we use a pre-trained ResNet18 (you might want a simpler model).
    surrogate_model = torchvision.models.resnet18(num_classes=10)
    surrogate_model = surrogate_model.to(device)
    surrogate_model.eval()  # we use it only for computing gradients in attack
    # Loss criterion for adversarial attack (using negative cross entropy to maximize loss)
    criterion = nn.CrossEntropyLoss()

    # For demonstration, generate a subset (e.g., first 500 images) of unlearnable examples.
    subset_size = 500
    base_subset = torch.utils.data.Subset(train_dataset, list(range(subset_size)))
    unlearnable_dataset = UnlearnableCIFAR10(
        base_subset, surrogate_model, criterion, steps=10, alpha=0.01, device=device
    )

    # Data loaders for training classifier on clean vs. unlearnable examples.
    batch_size = 32
    clean_train_loader = torch.utils.data.DataLoader(
        base_subset, batch_size=batch_size, shuffle=True
    )
    unlearnable_train_loader = torch.utils.data.DataLoader(
        unlearnable_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    num_epochs = 30
    print("Training classifier on clean data:")
    model_clean = train_classifier(
        clean_train_loader, test_loader, num_epochs=num_epochs, device=device
    )

    print("\nTraining classifier on unlearnable (perturbed) data:")
    model_unlearnable = train_classifier(
        unlearnable_train_loader, test_loader, num_epochs=num_epochs, device=device
    )

    # Here, you can compare the performance:
    # The expectation is that the classifier trained on unlearnable examples will show degraded accuracy.
    print("Performance on clean data:")
    test_classifier(model_clean, test_loader, device=device)

    print("Performance on unlearnable data:")
    test_classifier(model_unlearnable, test_loader, device=device)


if __name__ == "__main__":
    main()
