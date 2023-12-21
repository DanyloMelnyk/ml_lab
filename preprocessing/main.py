"""
Script for images processing
"""
import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.segmentation import flood
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)


def get_patient_id_from_filepath(filepath: Path) -> str:
    """Extract patient id from image filepath"""
    return filepath.name.split("_")[1]


def find_unique_patient_ids(dicoms_path: Path) -> List[str]:
    """Find unique patient ids from DICOM images in dicoms_path"""
    print("Search for files matching", dicoms_path / "*.dcm")
    dicoms_files = list(dicoms_path.glob("*.dcm"))

    matching_patient_ids = set(map(get_patient_id_from_filepath, dicoms_files))

    print(
        "Founded",
        len(dicoms_files),
        "images for",
        len(matching_patient_ids),
        "patients",
    )

    return list(matching_patient_ids)


def convert_dicom_data_to_ndarray(dicom_data: FileDataset) -> np.ndarray:
    """Convert DICOM data to ndarray"""
    if dicom_data.PhotometricInterpretation == "RGB":
        image = dicom_data.pixel_array
    elif dicom_data.PhotometricInterpretation == "MONOCHROME1":
        try:
            image = apply_voi_lut(dicom_data.pixel_array, dicom_data, prefer_lut=True)
        except ValueError:
            image = dicom_data.pixel_array
    else:
        try:
            image = apply_voi_lut(dicom_data.pixel_array, dicom_data, prefer_lut=False)
        except ValueError:
            image = dicom_data.pixel_array

    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        image = np.amax(image) - image

    image = image.astype(np.float64)

    # Rescale grey scale values to be between 0 and 65535
    image = (image / image.max()) * 65535.0

    image = image.astype(np.uint16)

    return image


def convert_dicom_to_png(dicom_image_path: Path, processed_image_path: Path):
    """Convert DICOM image to PNG"""
    dicom_data = pydicom.dcmread(dicom_image_path)

    image = convert_dicom_data_to_ndarray(dicom_data)
    cv2.imwrite(str(processed_image_path), image)


def process_patient_images(
    patient_id: str, dicoms_path: Path, images_path: Path, labels: pd.DataFrame
):
    """Process patient images"""
    patient_images = dicoms_path.glob(f"*_{patient_id}_*.dcm")
    patient_metadata = []

    for patient_image in patient_images:
        image_filename = patient_image.name
        image_id = int(image_filename.split("_")[0])
        image_label = labels[labels["File Name"] == image_id]

        if len(image_label) == 1:
            datetime_info = image_label["Acquisition date"].values[0]
            view = image_label["View"].values[0]
            side = image_label["Laterality"].values[0]
            density = image_label["ACR"].values[0]
            birads = image_label["Bi-Rads"].values[0]

            png_image_path = images_path / image_filename.replace(".dcm", ".png")

            convert_dicom_to_png(patient_image, png_image_path)

            metadata = {
                "study_datetime": datetime_info,
                "patient": patient_id,
                "ViewPosition": view,  # CC or MLO
                "side": side,
                "density": density,
                "birads": birads,
                "init_image_path": patient_image,
                "unprocessed_file_path": png_image_path,
            }

            patient_metadata.append(metadata)

        else:
            raise ValueError(f"Multiple rows with File Name '{image_id}'")

    return patient_metadata


def convert_dicoms_to_unprocessed_png(path: Path) -> pd.DataFrame:
    """Convert all dicoms to raw png, return metadata dataframe"""
    logging.info("Starting DICOM to PNG conversion...")
    dicoms_path = path / "AllDICOMs"
    labels_path = path / "INbreast.csv"

    labels = pd.read_csv(labels_path, sep=";").drop(
        columns=["Patient ID", "Patient age"]
    )

    images_path = path / "unprocessed_png"
    images_path.mkdir(exist_ok=True)

    processed_labels_path = path / "unprocessed_png_labels.csv"

    patients_ids = find_unique_patient_ids(dicoms_path)

    metadata = []

    for patient_id in tqdm(patients_ids):
        patient_metadata = process_patient_images(
            patient_id, dicoms_path, images_path, labels
        )
        metadata.extend(patient_metadata)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(processed_labels_path, index=False)

    logging.info("DICOM to PNG conversion completed.")

    return metadata_df


def morphological_analysis_cv_2(
    image: np.ndarray, se_size=15, iterations=5
) -> np.ndarray:
    """Perform morphological analysis on single input image."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))

    im_o = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
    for _ in range(iterations - 1):
        im_o = cv2.morphologyEx(im_o, cv2.MORPH_OPEN, se)

    im_c = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    for _ in range(iterations - 1):
        im_c = cv2.morphologyEx(im_c, cv2.MORPH_CLOSE, se)

    wth = cv2.bitwise_and(image, cv2.bitwise_not(im_o))

    bth = cv2.bitwise_and(im_c, cv2.bitwise_not(image))

    wth_or_bth = cv2.bitwise_or(bth, wth)
    not_wth_or_bth = cv2.bitwise_not(wth_or_bth).astype(np.uint8)

    return cv2.bitwise_and(image, image, mask=not_wth_or_bth)


def morphological_analysis(path: Path, metadata_df: pd.DataFrame) -> None:
    """Perform morphological analysis on unprocessed images from metadata_df."""
    logging.info("Starting morphological analysis...")
    output_images_path = path / "morphological_analysis_png"
    output_images_path.mkdir(exist_ok=True)

    def generate_ma_file_path(row: pd.Series) -> Path:
        input_img_path: Path = row["unprocessed_file_path"]

        img = cv2.imread(
            str(input_img_path),
            cv2.IMREAD_ANYDEPTH,
        )

        img_ = morphological_analysis_cv_2(img)
        cv2.imwrite(str(output_images_path / input_img_path.name), img_)

        return output_images_path / input_img_path.name

    metadata_df["ma_file_path"] = metadata_df.apply(generate_ma_file_path, axis=1)
    logging.info("Morphological analysis completed.")


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to the range [0, 1]"""
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def normalization(path: Path, metadata_df: pd.DataFrame):
    """Perform normalization on post morphological analysis images from metadata_df."""
    logging.info("Starting normalization...")
    output_images_path = path / "normalized_png"
    output_images_path.mkdir(exist_ok=True)

    def generate_normalized_file_path(row: pd.Series) -> Path:
        input_img_path: Path = row["ma_file_path"]

        img = cv2.imread(
            str(input_img_path),
            cv2.IMREAD_ANYDEPTH,
        )

        img_ = normalize_image(img)
        img_ = img_ * 65535.0
        img_ = img_.astype(np.uint16)

        # print(f"Normalized: {img_.min()=} {img_.max()=} {img_.dtype=}")
        cv2.imwrite(str(output_images_path / input_img_path.name), img_)

        return output_images_path / input_img_path.name

    metadata_df["normalized_file_path"] = metadata_df.apply(
        generate_normalized_file_path, axis=1
    )
    logging.info("Normalization completed.")


def region_based_segmentation(
    image: np.ndarray, seed_pixel: tuple[int, int], similarity_criteria: float
) -> np.ndarray:
    """Perform region based segmentation on single image."""
    mask = flood(image, seed_pixel, tolerance=similarity_criteria)

    return cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))


def segmentation(path: Path, metadata_df: pd.DataFrame):
    """Perform region based segmentation onnormalized images from metadata_df."""
    logging.info("Starting region-based segmentation...")
    output_images_path = path / "segmented_png"
    output_images_path.mkdir(exist_ok=True)

    def generate_segmented_file_path(row: pd.Series) -> Path:
        input_img_path: Path = row["normalized_file_path"]

        img = cv2.imread(
            str(input_img_path),
            cv2.IMREAD_ANYDEPTH,
        )

        h, w, *_ = img.shape
        if row["side"] == "L":
            seed_pixel = (h // 2, int(w * 1 / 3))
        else:
            seed_pixel = (h // 2, int(w * 2 / 3))

        # Similarity criteria for region-growing segmentation
        similarity_criteria = 25 * 257

        # Apply region-growing segmentation
        img_ = region_based_segmentation(img, seed_pixel, similarity_criteria)
        cv2.imwrite(str(output_images_path / input_img_path.name), img_)

        return output_images_path / input_img_path.name

    metadata_df["segmented_file_path"] = metadata_df.apply(
        generate_segmented_file_path, axis=1
    )
    logging.info("Region-based segmentation completed.")


def resize(path: Path, metadata_df: pd.DataFrame, size: tuple):
    logging.info("Starting resizing...")
    output_images_path = path / f"resized_{size[0]}x{size[1]}_png"
    output_images_path.mkdir(exist_ok=True)

    def generate_resized_file_path(row: pd.Series) -> Path:
        input_img_path: Path = row["segmented_file_path"]

        img = cv2.imread(
            str(input_img_path),
            cv2.IMREAD_GRAYSCALE,
        )

        img_ = cv2.resize(img, size)
        cv2.imwrite(str(output_images_path / input_img_path.name), img_)

        return output_images_path / input_img_path.name

    metadata_df["resized_file_path"] = metadata_df.apply(
        generate_resized_file_path, axis=1
    )
    logging.info("Resizing completed.")


def main(path: Path, size: tuple):
    metadata_df = convert_dicoms_to_unprocessed_png(path=path)

    # metadata_df = metadata_df.iloc[:5]

    morphological_analysis(path, metadata_df)

    normalization(path, metadata_df)

    segmentation(path, metadata_df)

    processed_labels_path = path / "processed_png_labels.csv"
    metadata_df.to_csv(processed_labels_path, index=False)

    resize(path, metadata_df, size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=Path, default=Path("../data/INbreast Release 1.0/")
    )
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    main(args.path, (args.height, args.width))
