# ============================================================
# SMALL CT + RTSTRUCT DATASET DOWNLOADER
#
# Dataset:
# LCTSC - Lung CT Segmentation Challenge 2017
#
# Downloads a small number of patients and converts:
#
# DICOM CT      -> NIfTI CT
# DICOM RTSTRUCT -> NIfTI lung mask
#
# Output:
#
# patient_data/
#     images/
#         patient_001.nii.gz
#         patient_002.nii.gz
#         patient_003.nii.gz
#
#     masks/
#         patient_001.nii.gz
#         patient_002.nii.gz
#         patient_003.nii.gz
#
# ============================================================

import os
import shutil
import tempfile

import SimpleITK as sitk

from huggingface_hub import snapshot_download
from rt_utils import RTStructBuilder


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = (
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "patient_data"
)

IMAGES_DIR = os.path.join(
    DATA_DIR,
    "images"
)

MASKS_DIR = os.path.join(
    DATA_DIR,
    "masks"
)

DOWNLOAD_DIR = os.path.join(
    PROJECT_DIR,
    "lctsc_download"
)


# ------------------------------------------------------------
# Number of patients
# ------------------------------------------------------------

NUMBER_OF_PATIENTS = 3


# ------------------------------------------------------------
# Hugging Face dataset
# ------------------------------------------------------------

DATASET = "MedOtter/LCTSC"


# ============================================================
# 2. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    IMAGES_DIR,
    exist_ok=True
)

os.makedirs(
    MASKS_DIR,
    exist_ok=True
)

os.makedirs(
    DOWNLOAD_DIR,
    exist_ok=True
)


print()
print("=" * 70)
print("SMALL CT DATASET")
print("=" * 70)

print()
print("Dataset:")
print(DATASET)

print()
print("Number of patients:")
print(NUMBER_OF_PATIENTS)

print()
print("Output:")
print(DATA_DIR)


# ============================================================
# 3. DOWNLOAD DATASET
# ============================================================

print()
print("=" * 70)
print("DOWNLOADING LCTSC DATA")
print("=" * 70)

print()
print(
    "Only the selected patient data will be used."
)

print()


# ------------------------------------------------------------
# Download the dataset repository
#
# The Hugging Face mirror contains the original DICOM
# CT + RTSTRUCT data.
# ------------------------------------------------------------

snapshot_download(
    repo_id=DATASET,
    repo_type="dataset",
    local_dir=DOWNLOAD_DIR
)


print()
print("Dataset download complete.")


# ============================================================
# 4. FIND PATIENT DIRECTORIES
# ============================================================

TRAIN_DIR = os.path.join(
    DOWNLOAD_DIR,
    "train"
)

if not os.path.exists(TRAIN_DIR):

    raise RuntimeError(
        "Could not find the LCTSC train directory:\n"
        + TRAIN_DIR
    )


patient_directories = sorted(
    [
        os.path.join(
            TRAIN_DIR,
            directory
        )
        for directory in os.listdir(TRAIN_DIR)
        if os.path.isdir(
            os.path.join(
                TRAIN_DIR,
                directory
            )
        )
    ]
)


if len(patient_directories) == 0:

    raise RuntimeError(
        "No patient directories were found."
    )


patient_directories = patient_directories[
    :NUMBER_OF_PATIENTS
]


print()
print("=" * 70)
print("PATIENTS SELECTED")
print("=" * 70)

for patient in patient_directories:

    print(
        os.path.basename(patient)
    )


# ============================================================
# 5. FIND DICOM FILES
# ============================================================

def find_dicom_files(folder):

    dicom_files = []

    for root, directories, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(".dcm"):

                dicom_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return dicom_files


# ============================================================
# 6. PROCESS EACH PATIENT
# ============================================================

for patient_number, patient_directory in enumerate(
    patient_directories,
    start=1
):

    patient_name = os.path.basename(
        patient_directory
    )

    print()
    print("=" * 70)
    print(
        f"PROCESSING PATIENT {patient_number}/"
        f"{len(patient_directories)}"
    )
    print("=" * 70)

    print()
    print("Patient:")
    print(patient_name)


    # --------------------------------------------------------
    # Find all DICOM files
    # --------------------------------------------------------

    dicom_files = find_dicom_files(
        patient_directory
    )

    if len(dicom_files) == 0:

        print(
            "No DICOM files found. Skipping."
        )

        continue


    # --------------------------------------------------------
    # Identify RTSTRUCT
    # --------------------------------------------------------

    rtstruct_file = None

    for filename in dicom_files:

        try:

            import pydicom

            dataset = pydicom.dcmread(
                filename,
                stop_before_pixels=True
            )

            if dataset.Modality == "RTSTRUCT":

                rtstruct_file = filename
                break

        except Exception:

            continue


    if rtstruct_file is None:

        print(
            "No RTSTRUCT found. Skipping."
        )

        continue


    print()
    print("RTSTRUCT:")
    print(rtstruct_file)


    # --------------------------------------------------------
    # Find CT files
    # --------------------------------------------------------

    ct_files = []

    for filename in dicom_files:

        try:

            dataset = pydicom.dcmread(
                filename,
                stop_before_pixels=True
            )

            if dataset.Modality == "CT":

                ct_files.append(
                    filename
                )

        except Exception:

            continue


    if len(ct_files) == 0:

        print(
            "No CT images found. Skipping."
        )

        continue


    print()
    print(
        "CT slices:",
        len(ct_files)
    )


    # ========================================================
    # 7. LOAD CT SERIES
    # ========================================================

    # --------------------------------------------------------
    # Find the CT series UID
    # --------------------------------------------------------

    series_uids = {}

    for filename in ct_files:

        try:

            dataset = pydicom.dcmread(
                filename,
                stop_before_pixels=True
            )

            uid = dataset.SeriesInstanceUID

            if uid not in series_uids:

                series_uids[uid] = []

            series_uids[uid].append(
                filename
            )

        except Exception:

            continue


    if len(series_uids) == 0:

        print(
            "Could not identify CT series."
        )

        continue


    # --------------------------------------------------------
    # Select largest CT series
    # --------------------------------------------------------

    ct_series_uid = max(
        series_uids,
        key=lambda uid:
        len(series_uids[uid])
    )

    ct_series_files = series_uids[
        ct_series_uid
    ]


    print()
    print(
        "Selected CT series:",
        len(ct_series_files),
        "slices"
    )


    # ========================================================
    # 8. COPY CT SERIES TO TEMPORARY DIRECTORY
    # ========================================================

    temporary_ct_dir = tempfile.mkdtemp()

    try:

        for filename in ct_series_files:

            shutil.copy2(
                filename,
                temporary_ct_dir
            )


        # ====================================================
        # 9. READ CT WITH SIMPLEITK
        # ====================================================

        reader = sitk.ImageSeriesReader()

        series_file_names = (
            reader.GetGDCMSeriesFileNames(
                temporary_ct_dir
            )
        )

        reader.SetFileNames(
            series_file_names
        )

        ct_image = reader.Execute()


        # ====================================================
        # 10. LOAD RTSTRUCT
        # ====================================================

        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=temporary_ct_dir,
            rt_struct_path=rtstruct_file
        )


        # ====================================================
        # 11. FIND LUNG STRUCTURES
        # ====================================================

        roi_names = rtstruct.get_roi_names()

        print()
        print("Available structures:")

        for roi in roi_names:

            print(
                "  ",
                roi
            )


        # ----------------------------------------------------
        # Look for left and right lung
        # ----------------------------------------------------

        left_lung = None
        right_lung = None

        for roi in roi_names:

            name = roi.lower()

            if (
                "lung_l" in name
                or "left lung" in name
                or "lung left" in name
            ):

                left_lung = roi

            if (
                "lung_r" in name
                or "right lung" in name
                or "lung right" in name
            ):

                right_lung = roi


        # ====================================================
        # 12. CREATE LUNG MASK
        # ====================================================

        lung_mask = None


        if left_lung is not None:

            print()
            print(
                "Using left lung:",
                left_lung
            )

            left_mask = rtstruct.get_roi_mask_by_name(
                left_lung
            )

            lung_mask = left_mask


        if right_lung is not None:

            print()
            print(
                "Using right lung:",
                right_lung
            )

            right_mask = rtstruct.get_roi_mask_by_name(
                right_lung
            )

            if lung_mask is None:

                lung_mask = right_mask

            else:

                lung_mask = (
                    lung_mask
                    | right_mask
                )


        # ----------------------------------------------------
        # Alternative: look for combined lung
        # ----------------------------------------------------

        if lung_mask is None:

            for roi in roi_names:

                name = roi.lower()

                if (
                    "lung" in name
                    and "left" not in name
                    and "right" not in name
                    and "lung_l" not in name
                    and "lung_r" not in name
                ):

                    print()
                    print(
                        "Using lung structure:",
                        roi
                    )

                    lung_mask = (
                        rtstruct.get_roi_mask_by_name(
                            roi
                        )
                    )

                    break


        if lung_mask is None:

            print()
            print(
                "Could not find a lung structure."
            )

            continue


        # ====================================================
        # 13. CONVERT MASK TO SIMPLEITK
        # ====================================================

        mask_array = (
            lung_mask.astype("uint8")
        )

        mask_image = sitk.GetImageFromArray(
            mask_array
        )

        mask_image.SetSpacing(
            ct_image.GetSpacing()
        )

        mask_image.SetOrigin(
            ct_image.GetOrigin()
        )

        mask_image.SetDirection(
            ct_image.GetDirection()
        )


        # ====================================================
        # 14. SAVE CT
        # ====================================================

        output_name = (
            f"patient_{patient_number:03d}.nii.gz"
        )

        ct_output = os.path.join(
            IMAGES_DIR,
            output_name
        )

        mask_output = os.path.join(
            MASKS_DIR,
            output_name
        )


        sitk.WriteImage(
            ct_image,
            ct_output
        )

        sitk.WriteImage(
            mask_image,
            mask_output
        )


        # ====================================================
        # 15. REPORT
        # ====================================================

        print()
        print(
            "CT saved:"
        )

        print(
            ct_output
        )

        print()
        print(
            "Mask saved:"
        )

        print(
            mask_output
        )


    finally:

        shutil.rmtree(
            temporary_ct_dir,
            ignore_errors=True
        )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("DOWNLOAD AND CONVERSION COMPLETE")
print("=" * 70)

print()

image_files = [
    f
    for f in os.listdir(IMAGES_DIR)
    if f.endswith(".nii.gz")
]

mask_files = [
    f
    for f in os.listdir(MASKS_DIR)
    if f.endswith(".nii.gz")
]

print(
    "CT volumes:",
    len(image_files)
)

print(
    "Segmentation masks:",
    len(mask_files)
)

print()
print("CT folder:")
print(IMAGES_DIR)

print()
print("Mask folder:")
print(MASKS_DIR)

print()
print("Your U-Net can now use:")
print()

print(
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
    r"\patient_data\images"
)

print()

print(
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
    r"\patient_data\masks"
)

print()
print("Done.")

