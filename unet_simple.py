import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Synthetic dataset
# ============================================================

class CircleDataset(Dataset):

    def __init__(self, n_samples=100, image_size=64):
        self.n_samples = n_samples
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        H = self.image_size
        W = self.image_size

        y, x = np.ogrid[:H, :W]

        # Random circle
        cx = np.random.randint(15, W - 15)
        cy = np.random.randint(15, H - 15)
        radius = np.random.randint(5, 12)

        mask = (
            (x - cx) ** 2 +
            (y - cy) ** 2
            <= radius ** 2
        )

        # Background noise
        image = np.random.normal(
            0,
            0.15,
            (H, W)
        )

        # Bright object
        image[mask] += 1.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return image, mask


# ============================================================
# 2. Double convolution block
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# 3. Small U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 8)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(8, 16)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(32, 16)

        self.up1 = nn.ConvTranspose2d(
            16,
            8,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(16, 8)

        # Final segmentation layer
        self.output = nn.Conv2d(
            8,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool3(e3)
        )

        # Decoder
        d3 = self.up3(b)

        # Skip connection
        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        # Skip connection
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        # Skip connection
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 4. Dice loss
# ============================================================

def dice_loss(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + 1e-6
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-6
    )

    return 1 - dice


# ============================================================
# 5. Dice metric
# ============================================================

def dice_score(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    prediction = prediction.view(-1)
    target = target.view(-1)

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + 1e-6
    ) / (
        prediction.sum()
        + target.sum()
        + 1e-6
    )

    return dice.item()


# ============================================================
# 6. Dataset
# ============================================================

train_dataset = CircleDataset(
    n_samples=100,
    image_size=64
)

val_dataset = CircleDataset(
    n_samples=20,
    image_size=64
)


train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=10,
    shuffle=False
)


# ============================================================
# 7. Model
# ============================================================

device = torch.device("cpu")

model = UNet().to(device)

print("\nModel created.")
print("Device:", device)

print(
    "Number of parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# 8. Optimizer
# ============================================================

learning_rate = 1e-3

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 9. Training
# ============================================================

n_epochs = 5

print("\nStarting training...\n")

for epoch in range(n_epochs):

    model.train()

    running_loss = 0.0

    total_batches = len(train_loader)

    for batch_idx, (images, masks) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)

        # Loss
        loss = dice_loss(
            predictions,
            masks
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Print iteration
        print(
            f"\rEpoch {epoch + 1}/{n_epochs} "
            f"| Batch {batch_idx}/{total_batches} "
            f"| Loss: {loss.item():.4f}",
            end=""
        )

    average_loss = (
        running_loss / total_batches
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    validation_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            predictions = model(images)

            validation_dice += dice_score(
                predictions,
                masks
            )

    validation_dice /= len(val_loader)

    print(
        f"\nEpoch {epoch + 1} complete "
        f"| Avg loss: {average_loss:.4f} "
        f"| Val Dice: {validation_dice:.4f}"
    )


# ============================================================
# 10. Show prediction
# ============================================================

model.eval()

image, mask = val_dataset[0]

with torch.no_grad():

    prediction = model(
        image.unsqueeze(0)
    )

    probability = torch.sigmoid(
        prediction
    )

    segmentation = (
        probability > 0.5
    ).float()


# ============================================================
# 11. Visualization
# ============================================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)

plt.imshow(
    image[0],
    cmap="gray"
)

plt.title("Input")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    mask[0],
    cmap="gray"
)

plt.title("Ground truth")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    segmentation[0, 0],
    cmap="gray"
)

plt.title("U-Net prediction")
plt.axis("off")


plt.tight_layout()

plt.show()