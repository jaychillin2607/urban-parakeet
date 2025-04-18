import os
import pandas as pd
import asyncio
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from src.config import config
from src.utils.logging import get_logger
from src.utils.file_io import read_csv_async
from src.utils.exceptions import DataError, async_exception_handler

logger = get_logger("data_preprocessing")


@async_exception_handler
async def load_annotation_files() -> pd.DataFrame:
    """
    Load and merge the two annotation files into a single DataFrame

    Returns:
        DataFrame with unified annotation format
    """
    try:
        logger.info("Loading annotation files...")

        # Load csv
        df1, df2_raw = await asyncio.gather(
            read_csv_async(config.data.statues_labels_file),
            read_csv_async(config.data.statues_labels2_file),
        )

        logger.debug(
            f"Loaded {len(df1)} entries from {config.data.statues_labels_file}"
        )

        logger.debug(
            f"Loaded {len(df2_raw)} entries from {config.data.statues_labels2_file}"
        )

        # Process second file (which has no headers)
        df2 = pd.DataFrame(df2_raw)
        if len(df2.columns) == 6:
            df2.columns = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
        else:
            logger.warning(
                f"Unexpected number of columns in {config.data.statues_labels2_file}"
            )
            raise DataError(f"Unexpected format in {config.data.statues_labels2_file}")

        # Convert class names to lowercase for consistency
        df1["class"] = df1["class"].str.lower()
        df2["class"] = df2["class"].str.lower()

        # Ensure numeric columns are numeric
        for col in ["xmin", "ymin", "xmax", "ymax"]:
            df1[col] = pd.to_numeric(df1[col])
            df2[col] = pd.to_numeric(df2[col])

        # Add temporary width and height for df2 images
        # We'll replace these with actual values later
        df2["width"] = 0
        df2["height"] = 0

        # Combine dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)

        # Map class names to integers
        mapped_values = combined_df["class"].map(config.data.class_map)

        if mapped_values.isna().any():
            logger.warning(f"Found {mapped_values.isna().sum()} unmapped class names")
            # Handle unmapped classes (e.g., assign to "other" class)
            mapped_values = mapped_values.fillna(config.data.class_map["other"])
        combined_df["class_id"] = mapped_values.astype(int)

        logger.info(f"Combined annotations with {len(combined_df)} entries")

        # Check for missing values
        missing_values = combined_df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(
                f"Found {missing_values} missing values in combined annotations"
            )

        # Some basic validation
        invalid_entries = combined_df[
            (combined_df["xmin"] >= combined_df["xmax"])
            | (combined_df["ymin"] >= combined_df["ymax"])
        ]

        if len(invalid_entries) > 0:
            logger.warning(f"Found {len(invalid_entries)} invalid bounding boxes")
            logger.debug(f"Invalid entries:\n{invalid_entries}")
            # Remove invalid entries
            combined_df = combined_df[
                (combined_df["xmin"] < combined_df["xmax"])
                & (combined_df["ymin"] < combined_df["ymax"])
            ]

        return combined_df

    except Exception as e:
        logger.error(f"Error loading annotation files: {str(e)}")
        raise DataError(f"Error loading annotation files: {str(e)}")


def find_image_path(filename: str) -> Optional[str]:
    """
    Find the full path for an image file by checking the class directories

    Args:
        filename: Image filename

    Returns:
        Full path to the image file or None if not found
    """
    # Check Lenin directory
    lenin_path = os.path.join(config.data.lenin_dir, filename)
    if os.path.exists(lenin_path):
        return lenin_path

    # Check Ataturk directory
    ataturk_path = os.path.join(config.data.ataturk_dir, filename)
    if os.path.exists(ataturk_path):
        return ataturk_path

    # Check Other directory
    other_path = os.path.join(config.data.other_dir, filename)
    if os.path.exists(other_path):
        return other_path

    # If not found, return None
    return None


async def get_image_dimensions(filename: str) -> Tuple[int, int]:
    """
    Get image dimensions (width, height)

    Args:
        filename: Image filename

    Returns:
        Tuple of (width, height)
    """
    try:
        image_path = find_image_path(filename)
        if image_path is None:
            logger.warning(f"Image file {filename} not found")
            return 0, 0

        # Use PIL's Image.open to extract dimensions without loading full image
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            img = await loop.run_in_executor(executor, lambda: Image.open(image_path))
            width, height = await loop.run_in_executor(executor, lambda: img.size)
            await loop.run_in_executor(executor, img.close)

        return width, height
    except Exception as e:
        logger.warning(f"Error getting dimensions for {filename}: {str(e)}")
        return 0, 0


@async_exception_handler
async def extract_image_dimensions(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Extract image dimensions for all unique images in the annotations

    Args:
        annotations: DataFrame with annotations

    Returns:
        Updated DataFrame with width and height
    """
    logger.info("Extracting image dimensions...")

    # Get unique filenames to avoid processing duplicates
    unique_files = annotations["filename"].unique()
    logger.info(f"Found {len(unique_files)} unique images")

    # Process images in parallel with a concurrency limit to avoid overwhelming the system
    MAX_CONCURRENCY = 50
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def get_dimensions_with_semaphore(filename):
        async with semaphore:
            return filename, await get_image_dimensions(filename)

    # Create tasks for all files
    tasks = [get_dimensions_with_semaphore(filename) for filename in unique_files]

    # Process files and show progress
    dimensions = {}
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        filename, (width, height) = await task

        dimensions[filename] = (width, height)
        if i % 100 == 0 or i == len(unique_files):
            logger.info(f"Processed {i}/{len(unique_files)} images")

    # Update the DataFrame with the extracted dimensions
    updated_annotations = annotations.copy()

    # Create a mapping of filename to dimensions for faster lookups
    for filename, (width, height) in dimensions.items():
        # Update all rows with this filename
        mask = updated_annotations["filename"] == filename
        updated_annotations.loc[mask, "width"] = width
        updated_annotations.loc[mask, "height"] = height

    # Check for missing dimensions
    missing_dims = updated_annotations[
        (updated_annotations["width"] == 0) | (updated_annotations["height"] == 0)
    ]
    if len(missing_dims) > 0:
        logger.warning(
            f"Could not extract dimensions for {len(missing_dims)} annotations"
        )
        updated_annotations = updated_annotations[
            (updated_annotations["width"] != 0) & (updated_annotations["height"] != 0)
        ]

    logger.info("Image dimensions extraction completed")
    return updated_annotations


@async_exception_handler
async def split_dataset(
    annotations: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets

    Args:
        annotations: DataFrame with annotations
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting dataset into train, validation, and test sets...")

    # Get unique filenames to avoid data leakage
    unique_files = annotations["filename"].unique()

    # First split for train and temp (val+test)
    train_files, temp_files = train_test_split(
        unique_files,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=None,  # We could add stratification by class if needed
    )

    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=random_state,
        shuffle=True,
    )

    # Create dataframes for each split
    train_df = annotations[annotations["filename"].isin(train_files)]
    val_df = annotations[annotations["filename"].isin(val_files)]
    test_df = annotations[annotations["filename"].isin(test_files)]

    # Check class distribution
    train_class_dist = train_df["class"].value_counts(normalize=True)
    val_class_dist = val_df["class"].value_counts(normalize=True)
    test_class_dist = test_df["class"].value_counts(normalize=True)

    logger.info(f"Train set: {len(train_files)} files, {len(train_df)} annotations")
    logger.info(f"Validation set: {len(val_files)} files, {len(val_df)} annotations")
    logger.info(f"Test set: {len(test_files)} files, {len(test_df)} annotations")

    logger.debug(f"Train class distribution: {train_class_dist.to_dict()}")
    logger.debug(f"Validation class distribution: {val_class_dist.to_dict()}")
    logger.debug(f"Test class distribution: {test_class_dist.to_dict()}")

    # Save split datasets
    os.makedirs(config.data.processed_data_dir, exist_ok=True)

    train_df.to_csv(config.data.train_annotations_file, index=False)
    val_df.to_csv(config.data.val_annotations_file, index=False)
    test_df.to_csv(config.data.test_annotations_file, index=False)

    logger.info("Dataset split and saved to processed data directory")

    return train_df, val_df, test_df


@async_exception_handler
async def preprocess_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function to prepare the dataset

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Starting dataset preprocessing...")

    # Load and merge annotation files
    combined_df = await load_annotation_files()

    # Extract image dimensions
    combined_df = await extract_image_dimensions(combined_df)

    # validate data
    combined_df = validate_boxes_in_dataframe(combined_df)

    # Split dataset
    train_df, val_df, test_df = await split_dataset(
        combined_df,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
    )

    logger.info("Dataset preprocessing completed")

    return train_df, val_df, test_df


def validate_boxes_in_dataframe(df, min_box_size=5):
    """
    Validate and fix bounding boxes in annotation DataFrame.

    Args:
        df: DataFrame with annotations
        min_box_size: Minimum box size in pixels

    Returns:
        DataFrame with validated/fixed annotations and count of issues
    """
    validated_df = df.copy()

    for idx, row in df.iterrows():
        width, height = row["width"], row["height"]

        # Skip rows with missing dimensions
        if width <= 0 or height <= 0:
            continue

        # Fix coordinates
        xmin = max(0, row["xmin"])
        ymin = max(0, row["ymin"])
        xmax = min(width, row["xmax"])
        ymax = min(height, row["ymax"])

        # Ensure minimum size
        if xmax - xmin < min_box_size:
            if xmax < width - min_box_size:
                xmax = xmin + min_box_size
            else:
                xmin = max(0, xmax - min_box_size)

        if ymax - ymin < min_box_size:
            if ymax < height - min_box_size:
                ymax = ymin + min_box_size
            else:
                ymin = max(0, ymax - min_box_size)

        # Ensure xmax > xmin and ymax > ymin
        if xmin >= xmax or ymin >= ymax:
            # Fix if possible
            if xmin >= xmax:
                xmax = min(width, xmin + min_box_size)
            if ymin >= ymax:
                ymax = min(height, ymin + min_box_size)

        # Update dataframe
        validated_df.loc[idx, "xmin"] = xmin
        validated_df.loc[idx, "ymin"] = ymin
        validated_df.loc[idx, "xmax"] = xmax
        validated_df.loc[idx, "ymax"] = ymax

    return validated_df


async def main():
    """Main function to run preprocessing"""
    await preprocess_dataset()


if __name__ == "__main__":
    asyncio.run(main())
