# ============================================================
# LCTSC: VISUALIZE ONE PATIENT AND ONE CT SLICE
#
# Displays:
#   - One CT slice
#   - RTSTRUCT contours for individual structures
#   - Different color for each structure
#   - Legend identifying each structure
# ============================================================

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pydicom
import SimpleITK as sitk

from rt_utils import RTStructBuilder


# ============================================================
# SETTINGS
# ============================================================

DATA_ROOT = (
    r"C:\Users\malyr\Documents\GitHub\unet-medical"
    r"\lctsc_download\train"
)

# ------------------------------------------------------------
# Select ONE patient
# ------------------------------------------------------------

PATIENT_ID = "LCTSC-Train-S1-001"

# ------------------------------------------------------------
# Select ONE CT slice
# ------------------------------------------------------------

SLICE_NUMBER = 50


# ============================================================
# STRUCTURES TO SHOW
# ============================================================
#
# None = show all structures
#
# Or specify explicitly:
#
# STRUCTURES_TO_SHOW = [
#     "SpinalCord",
#     "Lung_R",
#     "Lung_L",
#     "Heart",
#     "Esophagus"
# ]
#
# ============================================================

STRUCTURES_TO_SHOW = None


# ============================================================
# STRUCTURE COLORS
# ============================================================

STRUCTURE_COLORS = {

    "spinalcord": "magenta",

    "lung_r": "cyan",

    "lung_l": "lime",

    "heart": "red",

    "esophagus": "yellow",

}


# ============================================================
# FALLBACK COLORS
# ============================================================

FALLBACK_COLORS = [
    "magenta",
    "cyan",
    "lime",
    "red",
    "yellow",
    "orange",
    "blue",
    "white",
]


# ============================================================
# FIND PATIENT
# ============================================================

patient_directory = os.path.join(
    DATA_ROOT,
    PATIENT_ID
)


if not os.path.exists(
    patient_directory
):

    raise RuntimeError(
        "Patient directory does not exist:\n"
        + patient_directory
    )


print()
print("=" * 70)
print("LCTSC VISUALIZATION")
print("=" * 70)

print()
print("Patient:")
print(PATIENT_ID)

print()
print("Patient directory:")
print(patient_directory)


# ============================================================
# FIND DICOM FILES
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


dicom_files = find_dicom_files(
    patient_directory
)


print()
print(
    "DICOM files found:",
    len(dicom_files)
)


if len(dicom_files) == 0:

    raise RuntimeError(
        "No DICOM files found."
    )


# ============================================================
# FIND RTSTRUCT
# ============================================================

print()
print("Searching for RTSTRUCT...")


rtstruct_file = None


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

            rtstruct_file = filepath

            break

    except Exception:

        pass


if rtstruct_file is None:

    raise RuntimeError(
        "No RTSTRUCT found for patient "
        + PATIENT_ID
    )


print()
print("RTSTRUCT:")
print(rtstruct_file)


# ============================================================
# FIND CT SERIES
# ============================================================

print()
print("Searching for CT series...")


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

    raise RuntimeError(
        "No CT series found for patient "
        + PATIENT_ID
    )


# ============================================================
# SELECT LARGEST CT SERIES
# ============================================================

selected_series_uid = max(
    ct_series,
    key=lambda uid:
    len(
        ct_series[uid]["files"]
    )
)


selected_ct_series = ct_series[
    selected_series_uid
]


ct_directory = (
    selected_ct_series["directory"]
)


print()
print("CT directory:")
print(ct_directory)

print()
print(
    "CT slices:",
    len(
        selected_ct_series["files"]
    )
)


# ============================================================
# LOAD CT
# ============================================================

print()
print("Loading CT...")


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
)


print()
print(
    "CT volume shape:",
    ct_array.shape
)


print(
    "Number of CT slices:",
    ct_array.shape[0]
)


# ============================================================
# CHECK SLICE NUMBER
# ============================================================

number_of_slices = ct_array.shape[0]


if (
    SLICE_NUMBER < 0
    or SLICE_NUMBER >= number_of_slices
):

    raise ValueError(
        f"SLICE_NUMBER must be between "
        f"0 and {number_of_slices - 1}"
    )


print()
print(
    "Selected slice:",
    SLICE_NUMBER
)


# ============================================================
# LOAD RTSTRUCT
# ============================================================

print()
print("Loading RTSTRUCT...")


rtstruct = RTStructBuilder.create_from(
    dicom_series_path=ct_directory,
    rt_struct_path=rtstruct_file
)


# ============================================================
# GET STRUCTURE NAMES
# ============================================================

roi_names = rtstruct.get_roi_names()


print()
print("=" * 70)
print("STRUCTURES IN RTSTRUCT")
print("=" * 70)


for number, roi in enumerate(
    roi_names,
    start=1
):

    print(
        f"{number:2d}: {roi}"
    )


# ============================================================
# SELECT STRUCTURES
# ============================================================

if STRUCTURES_TO_SHOW is None:

    structures = roi_names

else:

    structures = []

    for requested_name in STRUCTURES_TO_SHOW:

        for roi in roi_names:

            if (
                roi.lower()
                == requested_name.lower()
            ):

                structures.append(
                    roi
                )

                break


print()
print("=" * 70)
print("STRUCTURES TO DISPLAY")
print("=" * 70)


for structure in structures:

    print(
        " ",
        structure
    )


# ============================================================
# GET CT SLICE
# ============================================================

ct_slice = ct_array[
    SLICE_NUMBER
]


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 10)
)


ax.imshow(
    ct_slice,
    cmap="gray"
)


# ============================================================
# OVERLAY STRUCTURES
# ============================================================

legend_handles = []

fallback_index = 0


for structure in structures:

    print()
    print(
        "Processing:",
        structure
    )


    # --------------------------------------------------------
    # Get mask
    # --------------------------------------------------------

    try:

        mask = (
            rtstruct.get_roi_mask_by_name(
                structure
            )
        )

    except Exception as error:

        print(
            "Could not create mask:",
            error
        )

        continue


    # --------------------------------------------------------
    # Convert X,Y,Z -> Z,Y,X
    # --------------------------------------------------------

    if mask.ndim != 3:

        print(
            "Unexpected mask dimensions:",
            mask.shape
        )

        continue


    mask = np.transpose(
        mask,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Check slice
    # --------------------------------------------------------

    if (
        SLICE_NUMBER
        >= mask.shape[0]
    ):

        print(
            "Slice outside mask volume."
        )

        continue


    mask_slice = mask[
        SLICE_NUMBER
    ]


    # --------------------------------------------------------
    # Check whether structure is present
    # --------------------------------------------------------

    if np.sum(mask_slice) == 0:

        print(
            "No contour on this slice."
        )

        continue


    # ========================================================
    # SELECT COLOR
    # ========================================================

    structure_key = (
        structure.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


    if structure_key in STRUCTURE_COLORS:

        color = STRUCTURE_COLORS[
            structure_key
        ]

    else:

        color = FALLBACK_COLORS[
            fallback_index
            % len(FALLBACK_COLORS)
        ]

        fallback_index += 1


    print(
        "Color:",
        color
    )


    # ========================================================
    # DRAW CONTOUR
    # ========================================================

    ax.contour(
        mask_slice.astype(float),
        levels=[0.5],
        colors=[color],
        linewidths=2.5
    )


    # ========================================================
    # CREATE LEGEND ENTRY
    # ========================================================
    #
    # We create the legend manually instead of using
    # contour.collections. This works with current versions
    # of Matplotlib.
    #
    # ========================================================

    legend_handle = Line2D(
        [0],
        [0],
        color=color,
        linewidth=3,
        label=structure
    )


    legend_handles.append(
        legend_handle
    )


# ============================================================
# LEGEND
# ============================================================

if len(legend_handles) > 0:

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        title="Structures",
        title_fontsize=12
    )


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    f"LCTSC\n"
    f"Patient: {PATIENT_ID}\n"
    f"CT slice: {SLICE_NUMBER}",
    fontsize=15
)


ax.axis("off")


# ============================================================
# SHOW
# ============================================================

plt.tight_layout()

plt.show()