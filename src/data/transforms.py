import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Callable, Dict, Any, Optional

from src.utils.logging import get_logger
from src.utils.exceptions import exception_handler

logger = get_logger("transforms")

@exception_handler
def get_transforms(
    image_size: Tuple[int, int] = (800, 800),
    use_augmentations: bool = True
) -> Tuple[Callable, Callable]:
    """
    Get transform functions for training and validation
    
    Args:
        image_size: Target image size (height, width)
        use_augmentations: Whether to use data augmentation
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    logger.info(f"Setting up transforms with image size {image_size}")
    
    # Validation transform (no augmentation)
    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )
    
    if not use_augmentations:
        logger.info("Using no augmentations for training")
        return val_transform, val_transform
    
    # Training transform with augmentations
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0
            ),
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RGBShift(p=0.5),
            ], p=0.7),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.5),
            
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.15, 0.15),
                    rotate=(-15, 15),
                    p=0.5
                ),
            ], p=0.5),
            
            # Normalization and conversion to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        )
    )
    
    logger.info("Created transforms with augmentations for training")
    
    return train_transform, val_transform