# ============================================================
# 2D U-NET LUNG SEGMENTATION USING LCTSC
#
# Workflow:
#   1. Find LCTSC patients
#   2. Load CT + RTSTRUCT
#   3. Build a binary lung mask from Lung_R + Lung_L
#   4. Split patients into train/validation sets
#   5. Train a 2D U-Net on ALL useful annotated slices
#   6. Predict a complete validation patient volume
#   7. Save the predicted mask as NIfTI
#   8. Save a comparison figure:
#        CT | Original | Prediction | Overlay
#   9. Report Dice score against the original RTSTRUCT
#
# The prediction is kept in the original CT volume dimensions
# so it can be directly compared with the original segmentation.
# ============================================================

import os
import glob
import random

import numpy as np
import matplotlib.pyplot as plt

import pydicom
import SimpleITK as sitk

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.transform import resize


# ============================================================
# 1. SETTINGS
# ============================================================


MODEL_PATH = os.path.join(
    r"C:\Users\malyr\Documents\GitHub\unet-medical",
    "model",
    "lung_unet.pth"
)

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

DATA_ROOT = (
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
    r"\lctsc_download\train"
)

OUTPUT_DIR = (
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
    r"\\predictions"
)

IMAGE_SIZE = 128

BATCH_SIZE = 8

N_EPOCHS = 22

LEARNING_RATE = 1e-3

RANDOM_SEED = 42

# If None, the script automatically uses the validation patient.
# You can also specify a patient explicitly, for example:
#
# PREDICT_PATIENT_ID = "LCTSC-Train-S1-003"
#
PREDICT_PATIENT_ID = None

# Combine right and left lung into one "lung tissue" class.
TARGET_STRUCTURES = [
    "Lung_R",
    "Lung_L",
]

# Only slices containing at least this many lung pixels are used.
# This removes the large number of empty slices outside the lungs.
MIN_LUNG_PIXELS = 10

# CT soft tissue window, matching the original script.
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. FIND DICOM FILES
# ============================================================

def find_dicom_files(folder):

    files_found = []

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(".dcm"):

                files_found.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return files_found


# ============================================================
# 5. FIND RTSTRUCT
# ============================================================

def find_rtstruct(dicom_files):

    for filepath in dicom_files:

        try:

            ds = pydicom.dcmread(
                filepath,
                stop_before_pixels=True
            )

            if getattr(
                ds,
                "Modality",
                ""
            ) == "RTSTRUCT":

                return filepath

        except Exception:

            pass

    return None


# ============================================================
# 6. FIND THE MAIN CT SERIES
# ============================================================

def find_ct_series(dicom_files):

    ct_series = {}

    for filepath in dicom_files:

        try:

            ds = pydicom.dcmread(
                filepath,
                stop_before_pixels=True
            )

            if getattr(
                ds,
                "Modality",
                ""
            ) != "CT":

                continue

            series_uid = getattr(
                ds,
                "SeriesInstanceUID",
                None
            )

            if series_uid is None:

                continue

            if series_uid not in ct_series:

                ct_series[series_uid] = {
                    "directory":
                        os.path.dirname(filepath),

                    "files": []
                }

            ct_series[
                series_uid
            ]["files"].append(
                filepath
            )

        except Exception:

            pass

    if len(ct_series) == 0:

        return None

    # Same strategy as visualize_lctsc.py:
    # choose the largest CT series.
    selected_uid = max(
        ct_series,
        key=lambda uid:
        len(
            ct_series[uid]["files"]
        )
    )

    return ct_series[selected_uid]


# ============================================================
# 7. LOAD ONE LCTSC PATIENT
# ============================================================

def load_patient(patient_id):

    patient_directory = os.path.join(
        DATA_ROOT,
        patient_id
    )

    if not os.path.exists(
        patient_directory
    ):

        raise RuntimeError(
            "Patient directory does not exist:\n"
            + patient_directory
        )

    dicom_files = find_dicom_files(
        patient_directory
    )

    if len(dicom_files) == 0:

        raise RuntimeError(
            "No DICOM files found for "
            + patient_id
        )

    rtstruct_file = find_rtstruct(
        dicom_files
    )

    if rtstruct_file is None:

        raise RuntimeError(
            "No RTSTRUCT found for "
            + patient_id
        )

    ct_series = find_ct_series(
        dicom_files
    )

    if ct_series is None:

        raise RuntimeError(
            "No CT series found for "
            + patient_id
        )

    ct_directory = (
        ct_series["directory"]
    )

    # --------------------------------------------------------
    # Load CT
    # --------------------------------------------------------

    reader = sitk.ImageSeriesReader()

    series_files = (
        reader.GetGDCMSeriesFileNames(
            ct_directory
        )
    )

    reader.SetFileNames(
        series_files
    )

    ct_image = reader.Execute()

    ct_array = sitk.GetArrayFromImage(
        ct_image
    ).astype(
        np.float32
    )

    # Shape:
    # Z x Y x X

    # --------------------------------------------------------
    # Load RTSTRUCT
    # --------------------------------------------------------

    from rt_utils import RTStructBuilder

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=ct_directory,
        rt_struct_path=rtstruct_file
    )

    roi_names = rtstruct.get_roi_names()

    # --------------------------------------------------------
    # Find right and left lung names
    # --------------------------------------------------------

    selected_rois = {}

    for target in TARGET_STRUCTURES:

        match = None

        for roi in roi_names:

            if roi.lower() == target.lower():

                match = roi

                break

        if match is None:

            # Try a slightly more tolerant match.
            target_key = (
                target.lower()
                .replace("_", "")
                .replace(" ", "")
                .replace("-", "")
            )

            for roi in roi_names:

                roi_key = (
                    roi.lower()
                    .replace("_", "")
                    .replace(" ", "")
                    .replace("-", "")
                )

                if roi_key == target_key:

                    match = roi

                    break

        if match is not None:

            selected_rois[target] = match

    if len(selected_rois) == 0:

        raise RuntimeError(
            "Could not find Lung_R or Lung_L "
            "in RTSTRUCT for "
            + patient_id
            + "\nAvailable structures:\n"
            + "\n".join(roi_names)
        )

    # --------------------------------------------------------
    # Build combined lung mask
    # --------------------------------------------------------

    lung_mask = np.zeros(
        ct_array.shape,
        dtype=np.uint8
    )

    for target, roi_name in selected_rois.items():

        print(
            f"  Loading {target}: {roi_name}"
        )

        roi_mask = rtstruct.get_roi_mask_by_name(
            roi_name
        )

        if roi_mask.ndim != 3:

            raise RuntimeError(
                "Unexpected RTSTRUCT mask dimensions: "
                + str(roi_mask.shape)
            )

        # rt-utils returns X x Y x Z.
        # Convert to Z x Y x X to match SimpleITK.
        roi_mask = np.transpose(
            roi_mask,
            (2, 0, 1)
        )

        if roi_mask.shape != ct_array.shape:

            raise RuntimeError(
                "Mask and CT dimensions do not match.\n"
                f"CT:   {ct_array.shape}\n"
                f"Mask: {roi_mask.shape}\n"
                f"Patient: {patient_id}"
            )

        lung_mask[
            roi_mask > 0
        ] = 1

    return {
        "patient_id": patient_id,
        "directory": patient_directory,
        "ct_directory": ct_directory,
        "rtstruct_file": rtstruct_file,
        "ct_image": ct_image,
        "ct_array": ct_array,
        "lung_mask": lung_mask,
        "roi_names": roi_names,
        "selected_rois": selected_rois,
    }


# ============================================================
# 8. DISCOVER PATIENTS
# ============================================================

patient_ids = sorted(
    [
        name
        for name in os.listdir(DATA_ROOT)
        if os.path.isdir(
            os.path.join(
                DATA_ROOT,
                name
            )
        )
        and name.lower().startswith(
            "lctsc-"
        )
    ]
)

if len(patient_ids) < 2:

    raise RuntimeError(
        "At least two LCTSC patients are required."
    )


print()
print("=" * 70)
print("LCTSC DATASET")
print("=" * 70)

print(
    "Data root:",
    DATA_ROOT
)

print(
    "Patients found:",
    len(patient_ids)
)

for patient_id in patient_ids:

    print(
        " ",
        patient_id
    )


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

train_ids, val_ids = train_test_split(
    patient_ids,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# With only a few patients, test_size=0.2 can produce
# a single validation patient. That is acceptable for
# this demonstration, but the validation result will be
# statistically weak.

print()
print("=" * 70)
print("PATIENT SPLIT")
print("=" * 70)

print(
    "Training patients:",
    len(train_ids)
)

for patient_id in train_ids:

    print(
        "  ",
        patient_id
    )

print()
print(
    "Validation patients:",
    len(val_ids)
)

for patient_id in val_ids:

    print(
        "  ",
        patient_id
    )


# ============================================================
# 10. LOAD TRAINING PATIENTS
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

train_patient_data = []

for patient_id in train_ids:

    print()
    print(
        "Loading:",
        patient_id
    )

    data = load_patient(
        patient_id
    )

    train_patient_data.append(
        data
    )

    print(
        "  CT shape:",
        data["ct_array"].shape
    )

    print(
        "  Lung pixels:",
        int(
            data["lung_mask"].sum()
        )
    )


# ============================================================
# 11. LOAD VALIDATION PATIENTS
# ============================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

val_patient_data = []

for patient_id in val_ids:

    print()
    print(
        "Loading:",
        patient_id
    )

    data = load_patient(
        patient_id
    )

    val_patient_data.append(
        data
    )

    print(
        "  CT shape:",
        data["ct_array"].shape
    )

    print(
        "  Lung pixels:",
        int(
            data["lung_mask"].sum()
        )
    )


# ============================================================
# 12. BUILD 2D SLICE INDEX
# ============================================================

def build_slice_index(patient_data):

    samples = []

    for patient in patient_data:

        mask = patient["lung_mask"]

        number_of_slices = mask.shape[0]

        for z in range(
            number_of_slices
        ):

            lung_pixels = int(
                mask[z].sum()
            )

            if lung_pixels >= MIN_LUNG_PIXELS:

                samples.append(
                    (
                        patient["patient_id"],
                        z
                    )
                )

    return samples


train_samples = build_slice_index(
    train_patient_data
)

val_samples = build_slice_index(
    val_patient_data
)

print()
print("=" * 70)
print("2D TRAINING SAMPLES")
print("=" * 70)

print(
    "Training slices:",
    len(train_samples)
)

print(
    "Validation slices:",
    len(val_samples)
)


# ============================================================
# 13. PATIENT LOOKUP
# ============================================================

patient_lookup = {}

for patient in (
    train_patient_data
    + val_patient_data
):

    patient_lookup[
        patient["patient_id"]
    ] = patient


# ============================================================
# 14. IMAGE PREPROCESSING
# ============================================================

def preprocess_ct(
    image
):

    lower = (
        WINDOW_CENTER
        - WINDOW_WIDTH / 2
    )

    upper = (
        WINDOW_CENTER
        + WINDOW_WIDTH / 2
    )

    image = np.clip(
        image,
        lower,
        upper
    )

    image = (
        image - lower
    ) / (
        upper - lower
    )

    image = resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        anti_aliasing=True,
        preserve_range=True
    )

    return image.astype(
        np.float32
    )


def preprocess_mask(
    mask
):

    mask = resize(
        mask.astype(np.float32),
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )

    return (
        mask > 0.5
    ).astype(
        np.float32
    )


# ============================================================
# 15. DATASET
# ============================================================

class LungSliceDataset(Dataset):

    def __init__(
        self,
        samples,
        patient_lookup
    ):

        self.samples = samples

        self.patient_lookup = (
            patient_lookup
        )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        patient_id, z = (
            self.samples[idx]
        )

        patient = self.patient_lookup[
            patient_id
        ]

        image = patient[
            "ct_array"
        ][z]

        mask = patient[
            "lung_mask"
        ][z]

        image = preprocess_ct(
            image
        )

        mask = preprocess_mask(
            mask
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ============================================================
# 16. DATALOADERS
# ============================================================

train_dataset = LungSliceDataset(
    train_samples,
    patient_lookup
)

val_dataset = LungSliceDataset(
    val_samples,
    patient_lookup
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 17. U-NET
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

            nn.BatchNorm2d(
                out_channels
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

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(
        self,
        x
    ):

        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            1,
            16
        )

        self.pool1 = nn.MaxPool2d(
            2
        )

        self.enc2 = DoubleConv(
            16,
            32
        )

        self.pool2 = nn.MaxPool2d(
            2
        )

        self.enc3 = DoubleConv(
            32,
            64
        )

        self.pool3 = nn.MaxPool2d(
            2
        )

        self.bottleneck = DoubleConv(
            64,
            128
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            128,
            64
        )

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            64,
            32
        )

        self.up1 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            32,
            16
        )

        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(
        self,
        x
    ):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(
            d1
        )

        return self.output(
            d1
        )


# ============================================================
# 18. LOSS FUNCTIONS
# ============================================================


def combined_loss(prediction, target, dice_weight=0.5):
    """
    Combined Dice + Binary Cross Entropy loss.
    """

    # BCE operates directly on logits
    bce = nn.functional.binary_cross_entropy_with_logits(
        prediction,
        target
    )

    # Convert logits to probabilities for Dice
    probability = torch.sigmoid(prediction)

    probability = probability.view(-1)
    target = target.view(-1)

    intersection = (probability * target).sum()

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        probability.sum()
        + target.sum()
        + 1e-6
    )

    dice_loss = 1.0 - dice

    return (
        dice_weight * dice_loss
        + (1.0 - dice_weight) * bce
    )

def dice_loss(
    logits,
    target
):

    probability = torch.sigmoid(
        logits
    )

    probability = probability.view(
        -1
    )

    target = target.view(
        -1
    )

    intersection = (
        probability * target
    ).sum()

    dice = (
        2 * intersection + 1e-6
    ) / (
        probability.sum()
        + target.sum()
        + 1e-6
    )

    return 1 - dice


def volume_difference_percent(
    prediction_mask,
    ground_truth_mask):
    """
    Calculate percentage difference in segmented volume.

    Positive = prediction is larger.
    Negative = prediction is smaller.
    """

    predicted_volume = np.sum(
        prediction_mask > 0
    )

    ground_truth_volume = np.sum(
        ground_truth_mask > 0
    )

    if ground_truth_volume == 0:
        return 0.0

    difference = (
        predicted_volume
        - ground_truth_volume
    )

    percentage = (
        difference
        / ground_truth_volume
    ) * 100.0

    return percentage

def precision_score(prediction, target):
    """
    Pixel-wise precision.

    Precision = TP / (TP + FP)

    Measures how much of the predicted lung
    is actually lung according to the ground truth.
    """

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    prediction = prediction.view(-1)
    target = target.view(-1)

    true_positive = (
        prediction * target
    ).sum()

    false_positive = (
        prediction * (1 - target)
    ).sum()

    precision = (
        true_positive + 1e-6
    ) / (
        true_positive
        + false_positive
        + 1e-6
    )

    return precision.item()

def dice_score(
    logits,
    target
):

    probability = torch.sigmoid(
        logits
    )

    prediction = (
        probability > 0.5
    ).float()

    prediction = prediction.view(
        -1
    )

    target = target.view(
        -1
    )

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
# 19. CREATE MODEL
# ============================================================

model = UNet().to(DEVICE)

n_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{n_parameters:,}"
)


# ============================================================
# 20. CHECK IF MODEL ALREADY EXISTS
# ============================================================

if os.path.exists(MODEL_PATH):

    print()
    print("=" * 70)
    print("TRAINED MODEL FOUND")
    print("=" * 70)

    print(
        "Loading trained model:"
    )

    print(
        MODEL_PATH
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print()
    print("Model loaded successfully.")
    print("Skipping training.")


else:

    print()
    print("=" * 70)
    print("NO TRAINED MODEL FOUND")
    print("=" * 70)

    print()
    print("Training U-Net from scratch...")


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    train_losses = []
    val_dices = []
    val_precision = []

    best_val_dice = -1.0
    best_val_precision = -1.0
    best_model_selection_score = -1.0

    best_model_path = os.path.join(
        OUTPUT_DIR,
        "best_lung_unet.pth"
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    plt.ion()

    fig_training, axes_training = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    loss_ax = axes_training[0]
    dice_ax = axes_training[1]


    for epoch in range(N_EPOCHS):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            # loss = dice_loss(
            #     predictions,
            #     masks
            # )
            
            loss = combined_loss(
                predictions,
                masks,
                dice_weight=0.5
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        average_loss = (
            running_loss
            / len(train_loader)
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_dice = 0.0
        validation_precision = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                predictions = model(images)

                validation_dice += dice_score(
                    predictions,
                    masks
                )
                
                validation_precision += precision_score(
                    predictions,
                    masks
                )
                
                # volume_difference = volume_difference_percent(
                #     predicted_volume_mask,
                #     original_volume_mask
                # )


        validation_dice /= len(val_loader)
        
        validation_precision /= len(val_loader)
        
        


        train_losses.append(
            average_loss
        )

        val_dices.append(
            validation_dice
        )
        
        
        val_precision.append(
            validation_precision
        )
        
        model_selection_score = (
            0.6 * validation_dice
            + 0.4 * validation_precision
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if model_selection_score > best_model_selection_score:
        
            best_model_selection_score = model_selection_score
        
            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"Epoch {epoch + 1:02d}/{N_EPOCHS} "
                f"| Loss: {average_loss:.5f} "
                f"| Validation Dice: {validation_dice:.5f} "
                f"| NEW BEST MODEL"
            )

        else:

            print(
                f"Epoch {epoch + 1:02d}/{N_EPOCHS} "
                f"| Loss: {average_loss:.5f} "
                f"| Validation Dice: {validation_dice:.5f}"
            )


        # ====================================================
        # UPDATE PLOTS
        # ====================================================

        loss_ax.clear()

        loss_ax.plot(
            range(
                1,
                len(train_losses) + 1
            ),
            train_losses,
            marker="o"
        )

        loss_ax.set_title(
            "Training loss"
        )

        loss_ax.set_xlabel(
            "Epoch"
        )

        loss_ax.set_ylabel(
            "Loss"
        )

        loss_ax.grid(
            True
        )


        dice_ax.clear()

        dice_ax.plot(
            range(
                1,
                len(val_dices) + 1
            ),
            val_dices,
            label="dice",
            color = 'C0',
            marker="o"
        )
        
        dice_ax.plot(
            range(
                1,
                len(val_precision) + 1
            ),
            val_precision,
            label="precision",
            color = 'C1',
            marker="x"
        )
        
        dice_ax.legend()
        
        dice_ax.set_title(
            "Validation metrics"
        )

        dice_ax.set_xlabel(
            "Epoch"
        )

        dice_ax.set_ylabel(
            "Score"
        )

        dice_ax.set_ylim(
            0,
            1.05
        )

        dice_ax.grid(
            True
        )


        fig_training.canvas.draw_idle()

        fig_training.canvas.flush_events()

        plt.pause(
            0.01
        )


    plt.ioff()


    # ========================================================
    # LOAD BEST MODEL AFTER TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation Dice:",
        f"{best_val_dice:.5f}"
    )

    print(
        "Model saved to:",
        MODEL_PATH
    )


    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print()
    print("Best model loaded for prediction.")


# ============================================================
# 22. SELECT PATIENT FOR FINAL PREDICTION
# ============================================================

if PREDICT_PATIENT_ID is None:

    prediction_patient_id = (
        val_ids[0]
    )

else:

    prediction_patient_id = (
        PREDICT_PATIENT_ID
    )

if prediction_patient_id not in patient_lookup:

    print()
    print(
        "Prediction patient was not part "
        "of the initial split."
    )

    print(
        "Loading patient separately:"
    )

    prediction_patient = load_patient(
        prediction_patient_id
    )

else:

    prediction_patient = patient_lookup[
        prediction_patient_id
    ]


print()
print("=" * 70)
print("FINAL PREDICTION")
print("=" * 70)

print(
    "Patient:",
    prediction_patient_id
)

print(
    "Original CT shape:",
    prediction_patient[
        "ct_array"
    ].shape
)


# ============================================================
# 23. PREDICT COMPLETE CT VOLUME
# ============================================================

ct_array = prediction_patient[
    "ct_array"
]

original_mask = prediction_patient[
    "lung_mask"
]

number_of_slices = ct_array.shape[0]

prediction_volume = np.zeros(
    ct_array.shape,
    dtype=np.uint8
)

probability_volume = np.zeros(
    ct_array.shape,
    dtype=np.float32
)

print()
print(
    "Predicting",
    number_of_slices,
    "CT slices..."
)

with torch.no_grad():

    for z in range(
        number_of_slices
    ):

        image = preprocess_ct(
            ct_array[z]
        )

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(
            0
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )

        logits = model(
            image_tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

        # ----------------------------------------------------
        # Resize prediction back to ORIGINAL CT resolution.
        #
        # This is important:
        # the U-Net works at IMAGE_SIZE x IMAGE_SIZE,
        # but the saved prediction has the original
        # 512 x 512 CT resolution.
        # ----------------------------------------------------

        probability_original = resize(
            probability,
            ct_array[z].shape,
            order=1,
            preserve_range=True,
            anti_aliasing=False
        )

        segmentation_original = (
            probability_original > 0.5
        ).astype(
            np.uint8
        )

        probability_volume[z] = (
            probability_original
        )

        prediction_volume[z] = (
            segmentation_original
        )


# ============================================================
# 24. VOLUME DICE
# ============================================================

def numpy_dice(
    prediction,
    target
):

    prediction = (
        prediction > 0
    )

    target = (
        target > 0
    )

    intersection = np.logical_and(
        prediction,
        target
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:

        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


volume_dice = numpy_dice(
    prediction_volume,
    original_mask
)

print()
print(
    "Complete-volume Dice:",
    f"{volume_dice:.5f}"
)

print(
    "Original lung voxels:",
    int(
        original_mask.sum()
    )
)

print(
    "Predicted lung voxels:",
    int(
        prediction_volume.sum()
    )
)


# ============================================================
# 25. SAVE PREDICTION AS NIFTI
# ============================================================

prediction_image = sitk.GetImageFromArray(
    prediction_volume.astype(
        np.uint8
    )
)

prediction_image.CopyInformation(
    prediction_patient[
        "ct_image"
    ]
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    prediction_patient_id
    + "_predicted_lungs.nii.gz"
)

sitk.WriteImage(
    prediction_image,
    prediction_path
)


# ------------------------------------------------------------
# Also save the original RTSTRUCT-derived lung mask.
# ------------------------------------------------------------

original_mask_image = (
    sitk.GetImageFromArray(
        original_mask.astype(
            np.uint8
        )
    )
)

original_mask_image.CopyInformation(
    prediction_patient[
        "ct_image"
    ]
)

original_mask_path = os.path.join(
    OUTPUT_DIR,
    prediction_patient_id
    + "_original_lungs.nii.gz"
)

sitk.WriteImage(
    original_mask_image,
    original_mask_path
)


# ============================================================
# 26. SAVE PROBABILITY VOLUME
# ============================================================

probability_image = sitk.GetImageFromArray(
    probability_volume.astype(
        np.float32
    )
)

probability_image.CopyInformation(
    prediction_patient[
        "ct_image"
    ]
)

probability_path = os.path.join(
    OUTPUT_DIR,
    prediction_patient_id
    + "_lung_probability.nii.gz"
)

sitk.WriteImage(
    probability_image,
    probability_path
)


print()
print(
    "Predicted mask saved to:"
)

print(
    prediction_path
)

print()
print(
    "Original mask saved to:"
)

print(
    original_mask_path
)

print()
print(
    "Probability volume saved to:"
)

print(
    probability_path
)


# ============================================================
# 27. CREATE SLICE-BY-SLICE COMPARISON
# ============================================================

# Select the slice where the ORIGINAL lungs have the largest
# cross-sectional area. This usually gives a useful visual
# comparison rather than an empty slice.

slice_scores = (
    original_mask
    .reshape(
        original_mask.shape[0],
        -1
    )
    .sum(
        axis=1
    )
)

comparison_slice = int(
    np.argmax(
        slice_scores
    )
)

ct_slice = ct_array[
    comparison_slice
]

original_slice = original_mask[
    comparison_slice
]

prediction_slice = prediction_volume[
    comparison_slice
]


# ============================================================
# 28. DISPLAY COMPARISON
# ============================================================

lower = (
    WINDOW_CENTER
    - WINDOW_WIDTH / 2
)

upper = (
    WINDOW_CENTER
    + WINDOW_WIDTH / 2
)

display_ct = np.clip(
    ct_slice,
    lower,
    upper
)


fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 5)
)


# ------------------------------------------------------------
# CT
# ------------------------------------------------------------

axes[0].imshow(
    display_ct,
    cmap="gray"
)

axes[0].set_title(
    "CT"
)


# ------------------------------------------------------------
# Original RTSTRUCT
# ------------------------------------------------------------

axes[1].imshow(
    display_ct,
    cmap="gray"
)

axes[1].imshow(
    np.ma.masked_where(
        original_slice == 0,
        original_slice
    ),
    alpha=0.45,
    cmap="Greens"
)

axes[1].contour(
    original_slice.astype(float),
    levels=[0.5],
    colors=["lime"],
    linewidths=2
)

axes[1].set_title(
    "Original RTSTRUCT"
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

axes[2].imshow(
    display_ct,
    cmap="gray"
)

axes[2].imshow(
    np.ma.masked_where(
        prediction_slice == 0,
        prediction_slice
    ),
    alpha=0.45,
    cmap="Reds"
)

axes[2].contour(
    prediction_slice.astype(float),
    levels=[0.5],
    colors=["red"],
    linewidths=2
)

axes[2].set_title(
    "U-Net prediction"
)


# ------------------------------------------------------------
# Overlay
# ------------------------------------------------------------

axes[3].imshow(
    display_ct,
    cmap="gray"
)

axes[3].contour(
    original_slice.astype(float),
    levels=[0.5],
    colors=["lime"],
    linewidths=2
)

axes[3].contour(
    prediction_slice.astype(float),
    levels=[0.5],
    colors=["red"],
    linewidths=2
)

axes[3].set_title(
    "Comparison\n"
    "Green = original | Red = prediction"
)


for ax in axes:

    ax.axis(
        "off"
    )


fig.suptitle(
    f"LCTSC Lung Segmentation\n"
    f"{prediction_patient_id} | "
    f"Slice {comparison_slice} | "
    f"Volume Dice = {volume_dice:.4f}",
    fontsize=15
)

plt.tight_layout()


comparison_path = os.path.join(
    OUTPUT_DIR,
    prediction_patient_id
    + "_comparison.png"
)

fig.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)


print()
print(
    "Comparison image saved to:"
)

print(
    comparison_path
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.txt"
)

# ============================================================
# 29. SAVE TRAINING HISTORY
# ============================================================

if "train_losses" in locals():



    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LCTSC LUNG U-NET TRAINING\n"
        )

        file.write(
            "=========================\n\n"
        )

        file.write(
            f"Image size: {IMAGE_SIZE}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Epochs: {N_EPOCHS}\n"
        )

        file.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        file.write(
            f"Device: {DEVICE}\n\n"
        )

        file.write(
            "Training patients:\n"
        )

        for patient_id in train_ids:

            file.write(
                f"  {patient_id}\n"
            )

        file.write(
            "\nValidation patients:\n"
        )

        for patient_id in val_ids:

            file.write(
                f"  {patient_id}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "Epoch | Training loss | Validation Dice | Validation Precision\n"
        )

        for i in range(
            len(train_losses)
        ):

            file.write(
                f"{i + 1:5d} | "
                f"{train_losses[i]:.6f} | "
                f"{val_dices[i]:.6f} | "
                f"{val_precision[i]:.6f}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            f"Best validation Dice: "
            f"{best_val_dice:.6f}\n"
        )

        file.write(
            f"Prediction patient: "
            f"{prediction_patient_id}\n"
        )

        file.write(
            f"Complete-volume Dice: "
            f"{volume_dice:.6f}\n"
        )

    print()
    print(
        "Training history saved to:"
    )

    print(
        history_path
    )

else:

    print()
    print(
        "Training history not available because "
        "the existing model was loaded."
    )


# ============================================================
# 30. FINAL OUTPUT SUMMARY
# ============================================================

print()
print("=" * 70)
print("WORKFLOW COMPLETE")
print("=" * 70)

print()
print(
    "1. Trained model:"
)

print(
    "   ",
    MODEL_PATH
)

print()
print(
    "2. Predicted lung mask:"
)

print(
    "   ",
    prediction_path
)

print()
print(
    "3. Original lung mask:"
)

print(
    "   ",
    original_mask_path
)

print()
print(
    "4. Probability volume:"
)

print(
    "   ",
    probability_path
)

print()
print(
    "5. Visual comparison:"
)

print(
    "   ",
    comparison_path
)

print()
print(
    "6. Training history:"
)

print(
    "   ",
    history_path
)

print()
print(
    f"Final volume Dice: {volume_dice:.5f}"
)

print()

plt.show()
