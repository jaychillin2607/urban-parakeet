import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Callable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import DataError, exception_handler
from src.data.transforms import get_transforms

logger = get_logger("dataset")


class StatueDataset(Dataset):
    """
    Dataset for statue detection and classification
    """

    def __init__(
        self,
        annotation_file: str,
        transform: Optional[Callable] = None,
        is_train: bool = True,
    ) -> None:
        """
        Initialize the dataset

        Args:
            annotation_file: Path to the annotation file
            transform: Optional transform to apply to images
            is_train: Whether this is a training dataset
        """
        self.annotation_file = annotation_file
        self.transform = transform
        self.is_train = is_train

        # Load annotations
        self.annotations = pd.read_csv(annotation_file)

        # Map class names to class IDs if not already present
        if "class_id" not in self.annotations.columns:
            self.annotations["class_id"] = self.annotations["class"].map(
                config.data.class_map
            )

        # Get unique image files
        self.image_files = self.annotations["filename"].unique()

        # Create a mapping from image file to its annotations
        self.file_to_annotations = self._create_annotation_mapping()

        logger.info(
            f"Loaded {len(self.image_files)} images with {len(self.annotations)} annotations"
        )

    def _create_annotation_mapping(self) -> Dict[str, pd.DataFrame]:
        """
        Create a mapping from image file to its annotations

        Returns:
            Dictionary mapping filenames to annotation dataframes
        """
        file_to_annotations = {}
        for filename in self.image_files:
            file_to_annotations[filename] = self.annotations[
                self.annotations["filename"] == filename
            ]

        return file_to_annotations

    def _find_image_path(self, filename: str) -> str:
        """
        Find the full path for an image file by checking the class directories

        Args:
            filename: Image filename

        Returns:
            Full path to the image file

        Raises:
            DataError: If the image file is not found
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

        # If not found, raise an error
        raise DataError(f"Image file {filename} not found in any class directory")

    @exception_handler
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - image: Tensor with the image
                - boxes: Tensor with bounding boxes [N, 4]
                - labels: Tensor with class labels [N]
                - image_id: Tensor with image ID
                - filename: Original filename
        """
        filename = self.image_files[idx]

        try:
            # Get the full path
            image_path = self._find_image_path(filename)

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Get annotations for this image
            image_annotations = self.file_to_annotations[filename]

            # Extract boxes and labels directly from dataframe
            boxes = image_annotations[["xmin", "ymin", "xmax", "ymax"]].values.astype( np.float32)
            labels = image_annotations["class_id"].values.astype(np.int64)

            # Apply transforms
            if self.transform:
                transformed = self.transform(
                    image=np.array(image), bboxes=boxes, labels=labels
                )
                image = transformed["image"]
                boxes = np.array(transformed["bboxes"], dtype=np.float32)
                labels = np.array(transformed["labels"], dtype=np.int64)
            else:
                # Convert PIL image to tensor
                image = T.ToTensor()(image)

            # If no boxes, add a dummy box
            if len(boxes) == 0:
                boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
                labels = np.array([0], dtype=np.int64)  # "other" class

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([idx])

            # Create target dictionary for Faster R-CNN
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
                "filename": filename,
            }

            return {"image": image, "target": target}

        except Exception as e:
            logger.error(f"Error loading image {filename}: {str(e)}")
            # Return a blank image and target as fallback
            image = torch.zeros((3, 800, 800), dtype=torch.float32)
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
            image_id = torch.tensor([idx])

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
                "filename": filename,
            }

            return {"image": image, "target": target}

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset

        Returns:
            Number of samples
        """
        return len(self.image_files)


def collate_fn(
    batch: List[Dict[str, Any]],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for DataLoader

    Args:
        batch: List of samples from the dataset

    Returns:
        Tuple of (images, targets)
    """
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    return images, targets


@exception_handler
def create_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating data loaders...")

    # Get transforms
    train_transform, val_transform = get_transforms(
        image_size=config.train.image_size,
        use_augmentations=config.train.use_augmentations,
    )

    # Create datasets
    train_dataset = StatueDataset(
        annotation_file=config.data.train_annotations_file,
        transform=train_transform,
        is_train=True,
    )

    val_dataset = StatueDataset(
        annotation_file=config.data.val_annotations_file,
        transform=val_transform,
        is_train=False,
    )

    test_dataset = StatueDataset(
        annotation_file=config.data.test_annotations_file,
        transform=val_transform,
        is_train=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.train.device == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.train.device == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.inference.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.inference.device == "cuda" else False,
    )

    logger.info(
        f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
