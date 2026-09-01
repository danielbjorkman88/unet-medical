import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Settings
# ============================================================

IMAGE_SIZE = 64

N_TRAIN = 100
N_VAL = 20

BATCH_SIZE = 10

N_EPOCHS = 5

LEARNING_RATE = 1e-3


# ============================================================
# 2. Synthetic dataset
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
# 3. Double convolution block
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
# 4. U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.enc1 = DoubleConv(1, 8)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(8, 16)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(32, 64)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Final segmentation layer
        # ----------------------------------------------------

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
# 5. Dice loss
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
# 6. Dice metric
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
# 7. Create datasets
# ============================================================

train_dataset = CircleDataset(
    n_samples=N_TRAIN,
    image_size=IMAGE_SIZE
)

val_dataset = CircleDataset(
    n_samples=N_VAL,
    image_size=IMAGE_SIZE
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 8. Device
# ============================================================

device = torch.device("cpu")

print()
print("=" * 70)
print("U-NET SEGMENTATION TRAINING")
print("=" * 70)

print()
print("Device:", device)


# ============================================================
# 9. Create model
# ============================================================

model = UNet().to(device)

n_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{n_parameters:,}"
)


# ============================================================
# 10. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. Fixed validation example
# ============================================================

# Because CircleDataset creates random data in __getitem__,
# we create one example once and keep it fixed.

val_image, val_mask = val_dataset[0]

val_image = val_image.to(device)
val_mask = val_mask.to(device)


# ============================================================
# 12. Training history
# ============================================================

epoch_numbers = []

epoch_train_losses = []
epoch_val_dices = []

batch_numbers = []
batch_losses = []


# ============================================================
# 13. Create LIVE plotting window
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(13, 8)
)

fig.canvas.manager.set_window_title(
    "U-Net Training Progress"
)

# ------------------------------------------------------------
# Top-left: training loss
# ------------------------------------------------------------

ax_loss = fig.add_subplot(2, 2, 1)

ax_loss.set_title(
    "Training Loss",
    fontsize=13
)

ax_loss.set_xlabel(
    "Training iteration"
)

ax_loss.set_ylabel(
    "Dice loss"
)

ax_loss.grid(True)

# ------------------------------------------------------------
# Top-right: validation Dice
# ------------------------------------------------------------

ax_dice = fig.add_subplot(2, 2, 2)

ax_dice.set_title(
    "Validation Dice",
    fontsize=13
)

ax_dice.set_xlabel(
    "Epoch"
)

ax_dice.set_ylabel(
    "Dice score"
)

ax_dice.set_ylim(
    0,
    1.05
)

ax_dice.grid(True)

# ------------------------------------------------------------
# Bottom-left: prediction
# ------------------------------------------------------------

ax_prediction = fig.add_subplot(2, 2, 3)

ax_prediction.set_title(
    "Current segmentation prediction",
    fontsize=13
)

ax_prediction.axis("off")

# ------------------------------------------------------------
# Bottom-right: status
# ------------------------------------------------------------

ax_status = fig.add_subplot(2, 2, 4)

ax_status.axis("off")


# ------------------------------------------------------------
# Show window BEFORE training starts
# ------------------------------------------------------------

plt.tight_layout()

plt.show(
    block=False
)

plt.pause(0.5)


# ============================================================
# 14. Training
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()


global_iteration = 0


for epoch in range(N_EPOCHS):

    model.train()

    running_loss = 0.0

    total_batches = len(train_loader)


    # ========================================================
    # Training batches
    # ========================================================

    for batch_idx, (images, masks) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        masks = masks.to(device)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predictions = model(images)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = dice_loss(
            predictions,
            masks
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Store loss
        # ----------------------------------------------------

        loss_value = loss.item()

        running_loss += loss_value

        global_iteration += 1

        batch_numbers.append(
            global_iteration
        )

        batch_losses.append(
            loss_value
        )


        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        print(
            f"\r"
            f"Epoch {epoch + 1}/{N_EPOCHS} "
            f"| Batch {batch_idx}/{total_batches} "
            f"| Iteration {global_iteration} "
            f"| Loss {loss_value:.4f}",
            end=""
        )


        # ====================================================
        # LIVE UPDATE OF TRAINING LOSS
        # ====================================================

        ax_loss.clear()

        ax_loss.plot(
            batch_numbers,
            batch_losses,
            linewidth=1
        )

        # Show current epoch average
        current_average = (
            running_loss / batch_idx
        )

        ax_loss.axhline(
            current_average,
            linestyle="--",
            linewidth=1
        )

        ax_loss.set_title(
            f"Training Loss | Epoch {epoch + 1}/{N_EPOCHS}"
        )

        ax_loss.set_xlabel(
            "Training iteration"
        )

        ax_loss.set_ylabel(
            "Dice loss"
        )

        ax_loss.grid(True)


        # ====================================================
        # LIVE STATUS
        # ====================================================

        ax_status.clear()
        ax_status.axis("off")

        current_lr = optimizer.param_groups[0]["lr"]

        status = (
            f"TRAINING STATUS\n\n"

            f"Epoch:              "
            f"{epoch + 1} / {N_EPOCHS}\n"

            f"Batch:              "
            f"{batch_idx} / {total_batches}\n"

            f"Global iteration:   "
            f"{global_iteration}\n\n"

            f"Current loss:       "
            f"{loss_value:.5f}\n"

            f"Epoch loss so far:  "
            f"{current_average:.5f}\n\n"

            f"Learning rate:      "
            f"{current_lr:.2e}\n"

            f"Batch size:         "
            f"{BATCH_SIZE}\n\n"

            f"Device:             "
            f"{device}\n\n"

            f"Parameters:         "
            f"{n_parameters:,}"
        )

        ax_status.text(
            0.05,
            0.95,
            status,
            transform=ax_status.transAxes,
            verticalalignment="top",
            fontsize=11,
            family="monospace"
        )


        # ====================================================
        # Refresh plot
        # ====================================================

        fig.suptitle(
            f"U-Net Training Progress    "
            f"Epoch {epoch + 1}/{N_EPOCHS}",
            fontsize=16
        )

        fig.tight_layout()

        fig.canvas.draw_idle()

        fig.canvas.flush_events()

        plt.pause(0.01)


    # ========================================================
    # End of epoch
    # ========================================================

    average_loss = (
        running_loss / total_batches
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    validation_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            validation_dice += dice_score(
                predictions,
                masks
            )

    validation_dice /= len(val_loader)


    # ========================================================
    # Store epoch results
    # ========================================================

    epoch_numbers.append(
        epoch + 1
    )

    epoch_train_losses.append(
        average_loss
    )

    epoch_val_dices.append(
        validation_dice
    )


    # ========================================================
    # Get prediction from fixed validation image
    # ========================================================

    with torch.no_grad():

        prediction = model(
            val_image.unsqueeze(0)
        )

        probability = torch.sigmoid(
            prediction
        )

        segmentation = (
            probability > 0.5
        ).float()


    # ========================================================
    # Update epoch loss plot
    # ========================================================

    ax_loss.clear()

    # Individual batch losses
    ax_loss.plot(
        batch_numbers,
        batch_losses,
        alpha=0.4,
        linewidth=1,
        label="Batch loss"
    )

    # Epoch-average loss
    epoch_iteration_positions = [
        (i + 1) * total_batches
        for i in range(len(epoch_train_losses))
    ]

    ax_loss.plot(
        epoch_iteration_positions,
        epoch_train_losses,
        marker="o",
        linewidth=2,
        label="Epoch average"
    )

    ax_loss.set_title(
        "Training Loss"
    )

    ax_loss.set_xlabel(
        "Training iteration"
    )

    ax_loss.set_ylabel(
        "Dice loss"
    )

    ax_loss.grid(True)

    ax_loss.legend()


    # ========================================================
    # Update validation Dice
    # ========================================================

    ax_dice.clear()

    ax_dice.plot(
        epoch_numbers,
        epoch_val_dices,
        marker="o",
        linewidth=2
    )

    ax_dice.set_title(
        "Validation Dice"
    )

    ax_dice.set_xlabel(
        "Epoch"
    )

    ax_dice.set_ylabel(
        "Dice score"
    )

    ax_dice.set_ylim(
        0,
        1.05
    )

    ax_dice.set_xticks(
        epoch_numbers
    )

    ax_dice.grid(True)


    # ========================================================
    # Update prediction
    # ========================================================

    ax_prediction.clear()

    ax_prediction.imshow(
        segmentation[0, 0].cpu(),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax_prediction.set_title(
        f"Prediction after epoch {epoch + 1}"
    )

    ax_prediction.axis("off")


    # ========================================================
    # Update status
    # ========================================================

    ax_status.clear()
    ax_status.axis("off")

    status = (
        f"EPOCH COMPLETE\n\n"

        f"Epoch:             "
        f"{epoch + 1} / {N_EPOCHS}\n\n"

        f"Training loss:     "
        f"{average_loss:.5f}\n\n"

        f"Validation Dice:   "
        f"{validation_dice:.5f}\n\n"

        f"Learning rate:     "
        f"{optimizer.param_groups[0]['lr']:.2e}\n\n"

        f"Total iterations:  "
        f"{global_iteration}\n\n"

        f"Device:            "
        f"{device}\n\n"

        f"Parameters:        "
        f"{n_parameters:,}"
    )

    ax_status.text(
        0.05,
        0.95,
        status,
        transform=ax_status.transAxes,
        verticalalignment="top",
        fontsize=11,
        family="monospace"
    )


    fig.suptitle(
        f"U-Net Training Progress    "
        f"Epoch {epoch + 1}/{N_EPOCHS}",
        fontsize=16
    )

    fig.tight_layout()

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.2)


    # ========================================================
    # Console epoch summary
    # ========================================================

    print()

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} complete "
        f"| Training loss: {average_loss:.5f} "
        f"| Validation Dice: {validation_dice:.5f}"
    )

    print()


# ============================================================
# 15. Training finished
# ============================================================

print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    f"Final training loss: "
    f"{epoch_train_losses[-1]:.5f}"
)

print(
    f"Final validation Dice: "
    f"{epoch_val_dices[-1]:.5f}"
)

print()


# ============================================================
# 16. Final visualization
# ============================================================

plt.ioff()

plt.show()