# ============================================================
# 2D U-NET SEGMENTATION OF PATIENT CT DATA
# ============================================================

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from skimage.transform import resize


# ============================================================
# 1. Settings
# ============================================================

IMAGE_SIZE = 64

BATCH_SIZE = 4

N_EPOCHS = 6

LEARNING_RATE = 1e-3

# Folder containing your patient data
DATA_DIR = r"C:\Users\malyr\Documents\GitHub\unet-medical\patient_data"


# ============================================================
# 2. Find patient files
# ============================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "images",
            "*.nii.gz"
        )
    )
)

mask_files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            "masks",
            "*.nii.gz"
        )
    )
)


print()
print("=" * 70)
print("PATIENT DATA")
print("=" * 70)

print(
    "CT images:",
    len(image_files)
)

print(
    "Masks:",
    len(mask_files)
)


# ------------------------------------------------------------
# Check that we actually found data
# ------------------------------------------------------------

if len(image_files) == 0:

    raise RuntimeError(
        "No CT images found. Check DATA_DIR."
    )


if len(mask_files) == 0:

    raise RuntimeError(
        "No masks found. Check DATA_DIR."
    )


if len(image_files) != len(mask_files):

    raise RuntimeError(
        "Number of CT images and masks do not match."
    )


# ------------------------------------------------------------
# Display first few files
# ------------------------------------------------------------

print()

for image_file, mask_file in zip(
    image_files[:5],
    mask_files[:5]
):

    print(
        "CT:  ",
        os.path.basename(image_file)
    )

    print(
        "Mask:",
        os.path.basename(mask_file)
    )

    print()


# ============================================================
# 3. Patient CT Dataset
# ============================================================

class CT2DDataset(Dataset):

    def __init__(
        self,
        image_files,
        mask_files,
        image_size=64
    ):

        self.image_files = image_files

        self.mask_files = mask_files

        self.image_size = image_size


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, idx):

        # ====================================================
        # Load 3D CT volume
        # ====================================================

        ct = nib.load(
            self.image_files[idx]
        ).get_fdata()


        # ====================================================
        # Load 3D segmentation mask
        # ====================================================

        mask = nib.load(
            self.mask_files[idx]
        ).get_fdata()


        # ====================================================
        # Check dimensions
        # ====================================================

        if ct.shape != mask.shape:

            raise RuntimeError(
                f"CT and mask dimensions do not match:\n"
                f"CT: {ct.shape}\n"
                f"Mask: {mask.shape}"
            )


        # ====================================================
        # Find slices containing the structure
        # ====================================================

        valid_slices = []

        for z in range(mask.shape[2]):

            if np.sum(mask[:, :, z]) > 0:

                valid_slices.append(z)


        # ====================================================
        # If no slice contains the structure
        # ====================================================

        if len(valid_slices) == 0:

            # Fall back to middle slice
            slice_number = mask.shape[2] // 2

        else:

            # Randomly select a slice containing the target
            slice_number = np.random.choice(
                valid_slices
            )


        # ====================================================
        # Extract 2D slice
        # ====================================================

        image = ct[:, :, slice_number]

        mask = mask[:, :, slice_number]


        # ====================================================
        # CT WINDOWING
        #
        # Soft tissue window
        #
        # Center = 40 HU
        # Width  = 400 HU
        # ====================================================

        window_center = 40

        window_width = 400

        lower = (
            window_center
            - window_width / 2
        )

        upper = (
            window_center
            + window_width / 2
        )


        image = np.clip(
            image,
            lower,
            upper
        )


        # ====================================================
        # Normalize CT to 0-1
        # ====================================================

        image = (
            image - lower
        ) / (
            upper - lower
        )


        # ====================================================
        # Resize CT image
        # ====================================================

        image = resize(
            image,
            (
                self.image_size,
                self.image_size
            ),
            anti_aliasing=True
        )


        # ====================================================
        # Resize segmentation mask
        #
        # IMPORTANT:
        # order=0 means nearest-neighbour interpolation.
        #
        # We do NOT want interpolation creating values like
        # 0.37 or 0.62 in a binary segmentation mask.
        # ====================================================

        mask = resize(
            mask,
            (
                self.image_size,
                self.image_size
            ),
            order=0,
            preserve_range=True
        )


        # ====================================================
        # Convert mask to binary
        # ====================================================

        mask = (
            mask > 0
        ).astype(
            np.float32
        )


        # ====================================================
        # Convert image to PyTorch tensor
        #
        # From:
        #
        #     H × W
        #
        # To:
        #
        #     1 × H × W
        #
        # ====================================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)


        # ====================================================
        # Convert mask to PyTorch tensor
        # ====================================================

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)


        return image, mask


# ============================================================
# 4. Train / validation split
# ============================================================

train_images, val_images, train_masks, val_masks = (
    train_test_split(
        image_files,
        mask_files,
        test_size=0.2,
        random_state=42
    )
)


print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training patients:",
    len(train_images)
)

print(
    "Validation patients:",
    len(val_images)
)

print()


# ============================================================
# 5. Create datasets
# ============================================================

train_dataset = CT2DDataset(
    train_images,
    train_masks,
    image_size=IMAGE_SIZE
)


val_dataset = CT2DDataset(
    val_images,
    val_masks,
    image_size=IMAGE_SIZE
)


# ============================================================
# 6. Create DataLoaders
# ============================================================

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
# 7. Double convolution block
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# 8. U-Net
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ====================================================
        # Encoder
        # ====================================================

        self.enc1 = DoubleConv(
            1,
            8
        )

        self.pool1 = nn.MaxPool2d(2)


        self.enc2 = DoubleConv(
            8,
            16
        )

        self.pool2 = nn.MaxPool2d(2)


        self.enc3 = DoubleConv(
            16,
            32
        )

        self.pool3 = nn.MaxPool2d(2)


        # ====================================================
        # Bottleneck
        # ====================================================

        self.bottleneck = DoubleConv(
            32,
            64
        )


        # ====================================================
        # Decoder
        # ====================================================

        self.up3 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            64,
            32
        )


        self.up2 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            32,
            16
        )


        self.up1 = nn.ConvTranspose2d(
            16,
            8,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            16,
            8
        )


        # ====================================================
        # Final segmentation layer
        # ====================================================

        self.output = nn.Conv2d(
            8,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # ====================================================
        # Encoder
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )


        # ====================================================
        # Bottleneck
        # ====================================================

        b = self.bottleneck(
            self.pool3(e3)
        )


        # ====================================================
        # Decoder
        # ====================================================

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
# 9. Dice loss
# ============================================================

def dice_loss(
    prediction,
    target
):

    prediction = torch.sigmoid(
        prediction
    )

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
# 10. Dice metric
# ============================================================

def dice_score(
    prediction,
    target
):

    prediction = torch.sigmoid(
        prediction
    )


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
# 11. Device
# ============================================================

device = torch.device(
    "cpu"
)


print("=" * 70)
print("U-NET")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# 12. Create model
# ============================================================

model = UNet().to(
    device
)


n_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "Parameters:",
    f"{n_parameters:,}"
)


# ============================================================
# 13. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 14. Fixed validation example
# ============================================================

val_image, val_mask = val_dataset[0]

val_image = val_image.to(
    device
)

val_mask = val_mask.to(
    device
)


# ============================================================
# 15. Training history
# ============================================================

epoch_numbers = []

epoch_train_losses = []

epoch_val_dices = []

batch_numbers = []

batch_losses = []


# ============================================================
# 16. Plotting
# ============================================================

plt.ion()


# ------------------------------------------------------------
# Training loss window
# ------------------------------------------------------------

loss_fig, loss_ax = plt.subplots(
    figsize=(10, 6)
)

loss_fig.canvas.manager.set_window_title(
    "U-Net Training Loss"
)

loss_ax.set_title(
    "Training Loss",
    fontsize=16
)

loss_ax.set_xlabel(
    "Training iteration"
)

loss_ax.set_ylabel(
    "Dice loss"
)

loss_ax.grid(True)


# ------------------------------------------------------------
# Validation Dice window
# ------------------------------------------------------------

dice_fig, dice_ax = plt.subplots(
    figsize=(10, 6)
)

dice_fig.canvas.manager.set_window_title(
    "U-Net Validation Dice"
)

dice_ax.set_title(
    "Validation Dice",
    fontsize=16
)

dice_ax.set_xlabel(
    "Epoch"
)

dice_ax.set_ylabel(
    "Dice score"
)

dice_ax.set_ylim(
    0,
    1.05
)

dice_ax.grid(True)


# ------------------------------------------------------------
# Segmentation window
# ------------------------------------------------------------

prediction_fig, prediction_axes = plt.subplots(
    1,
    3,
    figsize=(12, 4)
)

prediction_fig.canvas.manager.set_window_title(
    "U-Net Segmentation"
)

prediction_axes[0].set_title(
    "CT"
)

prediction_axes[1].set_title(
    "Ground Truth"
)

prediction_axes[2].set_title(
    "Prediction"
)

for ax in prediction_axes:

    ax.axis("off")


plt.show(
    block=False
)

plt.pause(
    0.5
)


# ============================================================
# 17. Training
# ============================================================

global_iteration = 0


print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()


for epoch in range(
    N_EPOCHS
):

    model.train()

    running_loss = 0.0

    total_batches = len(
        train_loader
    )


    # ========================================================
    # Training batches
    # ========================================================

    for batch_idx, (
        images,
        masks
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(
            device
        )

        masks = masks.to(
            device
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        predictions = model(
            images
        )


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
        # Record
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


        # ----------------------------------------------------
        # Update loss plot
        # ----------------------------------------------------

        loss_ax.clear()

        loss_ax.plot(
            batch_numbers,
            batch_losses,
            linewidth=1
        )

        loss_ax.set_title(
            f"Training Loss | "
            f"Epoch {epoch + 1}/{N_EPOCHS}"
        )

        loss_ax.set_xlabel(
            "Training iteration"
        )

        loss_ax.set_ylabel(
            "Dice loss"
        )

        loss_ax.grid(True)

        loss_fig.canvas.draw_idle()

        loss_fig.canvas.flush_events()

        plt.pause(
            0.01
        )


    # ========================================================
    # End of epoch
    # ========================================================

    average_loss = (
        running_loss
        / total_batches
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    validation_dice = 0.0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )


            predictions = model(
                images
            )


            validation_dice += dice_score(
                predictions,
                masks
            )


    validation_dice /= len(
        val_loader
    )


    # ========================================================
    # Store history
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
    # Update Dice plot
    # ========================================================

    dice_ax.clear()

    dice_ax.plot(
        epoch_numbers,
        epoch_val_dices,
        marker="o",
        linewidth=2
    )

    dice_ax.set_title(
        f"Validation Dice | "
        f"Epoch {epoch + 1}/{N_EPOCHS}"
    )

    dice_ax.set_xlabel(
        "Epoch"
    )

    dice_ax.set_ylabel(
        "Dice score"
    )

    dice_ax.set_ylim(
        0,
        1.05
    )

    dice_ax.set_xticks(
        epoch_numbers
    )

    dice_ax.grid(True)

    dice_fig.canvas.draw_idle()

    dice_fig.canvas.flush_events()


    # ========================================================
    # Current segmentation prediction
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
    # Update segmentation visualization
    # ========================================================

    prediction_axes[0].clear()

    prediction_axes[1].clear()

    prediction_axes[2].clear()


    prediction_axes[0].imshow(
        val_image[0].cpu(),
        cmap="gray"
    )

    prediction_axes[0].set_title(
        "CT"
    )


    prediction_axes[1].imshow(
        val_mask[0].cpu(),
        cmap="gray"
    )

    prediction_axes[1].set_title(
        "Ground Truth"
    )


    prediction_axes[2].imshow(
        segmentation[0, 0].cpu(),
        cmap="gray"
    )

    prediction_axes[2].set_title(
        f"Prediction\nEpoch {epoch + 1}"
    )


    for ax in prediction_axes:

        ax.axis("off")


    prediction_fig.canvas.draw_idle()

    prediction_fig.canvas.flush_events()


    plt.pause(
        0.1
    )


    # ========================================================
    # Console summary
    # ========================================================

    print()

    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} complete "
        f"| Training loss: {average_loss:.5f} "
        f"| Validation Dice: {validation_dice:.5f}"
    )

    print()


# ============================================================
# 18. Training finished
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
# 19. Keep plots open
# ============================================================

plt.ioff()

plt.show()