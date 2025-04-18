import os
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

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
        
        # Load statues_labels.csv
        df1 = await read_csv_async(config.data.statues_labels_file)
        logger.debug(f"Loaded {len(df1)} entries from {config.data.statues_labels_file}")
        
        # Load statues_labels2.csv
        df2_raw = await read_csv_async(config.data.statues_labels2_file)
        logger.debug(f"Loaded {len(df2_raw)} entries from {config.data.statues_labels2_file}")
        
        # Process second file (which has no headers)
        df2 = pd.DataFrame(df2_raw)
        if len(df2.columns) == 6:
            df2.columns = ["filename", "xmin", "ymin", "xmax", "ymax", "class"]
        else:
            logger.warning(f"Unexpected number of columns in {config.data.statues_labels2_file}")
            raise DataError(f"Unexpected format in {config.data.statues_labels2_file}")
        
        # Convert class names to lowercase for consistency
        df1["class"] = df1["class"].str.lower()
        df2["class"] = df2["class"].str.lower()
        
        # Ensure numeric columns are numeric
        for col in ["xmin", "ymin", "xmax", "ymax"]:
            df1[col] = pd.to_numeric(df1[col])
            df2[col] = pd.to_numeric(df2[col])
        
        # Add width and height for df2 images
        # We'll replace these with actual values when loading images
        df2["width"] = 0
        df2["height"] = 0
        
        # Combine dataframes
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Map class names to integers
        combined_df["class_id"] = combined_df["class"].map(config.data.class_map)
        
        logger.info(f"Combined annotations with {len(combined_df)} entries")
        
        # Check for missing values
        missing_values = combined_df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in combined annotations")
        
        # Some basic validation
        invalid_entries = combined_df[
            (combined_df["xmin"] >= combined_df["xmax"]) | 
            (combined_df["ymin"] >= combined_df["ymax"])
        ]
        
        if len(invalid_entries) > 0:
            logger.warning(f"Found {len(invalid_entries)} invalid bounding boxes")
            logger.debug(f"Invalid entries:\n{invalid_entries}")
            # Remove invalid entries
            combined_df = combined_df[
                (combined_df["xmin"] < combined_df["xmax"]) & 
                (combined_df["ymin"] < combined_df["ymax"])
            ]
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading annotation files: {str(e)}")
        raise DataError(f"Error loading annotation files: {str(e)}")

@async_exception_handler
async def split_dataset(
    annotations: pd.DataFrame, 
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
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
        shuffle=True
    )
    
    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=random_state,
        shuffle=True
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
    
    # Split dataset
    train_df, val_df, test_df = await split_dataset(
        combined_df,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split
    )
    
    logger.info("Dataset preprocessing completed")
    
    return train_df, val_df, test_df

async def main():
    """Main function to run preprocessing"""
    await preprocess_dataset()

if __name__ == "__main__":
    asyncio.run(main())