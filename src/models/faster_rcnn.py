import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, ResNet152_Weights
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import ModelError, exception_handler

logger = get_logger("faster_rcnn")

@exception_handler
def create_faster_rcnn_model(
num_classes: int = 4,  # Updated for background + 3 classes 
    backbone: str = "resnet50",
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 5,
    box_score_thresh: float = 0.05,  # Lowered from 0.5 to 0.05
    box_nms_thresh: float = 0.3,
    box_detections_per_img: int = 100
) -> FasterRCNN:
    """
    Create a Faster R-CNN model with custom backbone and number of classes
    
    Args:
        num_classes: Number of target classes (including background)
        backbone: Backbone architecture to use (resnet50, resnet101, etc.)
        pretrained: Whether to use pretrained weights for the full model
        pretrained_backbone: Whether to use pretrained weights for the backbone
        trainable_backbone_layers: Number of backbone layers to make trainable
        box_score_thresh: Threshold for box score
        box_nms_thresh: Threshold for box NMS
        box_detections_per_img: Maximum number of detections per image
        
    Returns:
        Configured Faster R-CNN model
    """
    logger.info(f"Creating Faster R-CNN model with {backbone} backbone")
    
    # For compatibility with torchvision API
    if backbone not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        raise ModelError(f"Unsupported backbone: {backbone}. Choose from resnet18/34/50/101/152")
    
    # Get appropriate weights enum based on backbone
    weights = None
    if pretrained_backbone:
        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT
        elif backbone == "resnet34":
            weights = ResNet34_Weights.DEFAULT
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT
        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT
        elif backbone == "resnet152":
            weights = ResNet152_Weights.DEFAULT
    
    # Create backbone
    backbone_network = resnet_fpn_backbone(
        backbone_name=backbone,
        weights=weights,  # Modern way to specify pretrained weights
        trainable_layers=trainable_backbone_layers
    )
    
    # Create Faster R-CNN model
    model = FasterRCNN(
        backbone=backbone_network,
        num_classes=91,  # COCO classes
        min_size=800,
        max_size=1333,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img
    )
    
    if pretrained:
        logger.info("Loading pretrained weights")
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    logger.info(f"Created Faster R-CNN model with {num_classes} classes")
    
    return model

@exception_handler
def save_model(
    model: nn.Module, 
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    accuracy: Optional[float] = None
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        filepath: Path to save the model
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        loss: Optional loss value
        accuracy: Optional accuracy value
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    if accuracy is not None:
        checkpoint["accuracy"] = accuracy
    
    torch.save(checkpoint, filepath)
    logger.info(f"Model saved to {filepath}")

@exception_handler
def load_model(
    filepath: str,
    device: str = "cuda",
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load model from checkpoint
    
    Args:
        filepath: Path to the model checkpoint
        device: Device to load the model to
        optimizer: Optional optimizer to load state into
        
    Returns:
        Tuple of (model, optimizer, metadata)
    """
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, using CPU instead")
        device = "cpu"
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create a new model
    model = create_faster_rcnn_model(
        num_classes=config.model.num_classes,
        backbone=config.model.backbone,
        pretrained=False,
        box_score_thresh=config.model.box_score_thresh,
        box_nms_thresh=config.model.box_nms_thresh,
        box_detections_per_img=config.model.box_detections_per_img
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Load optimizer if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Extract metadata
    metadata = {}
    for key in ["epoch", "loss", "accuracy"]:
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    
    logger.info(f"Model loaded from {filepath}")
    return model, optimizer, metadata