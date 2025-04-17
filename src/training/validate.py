import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import os

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import exception_handler
from src.utils.file_io import read_image_async, save_image_async

logger = get_logger("validation")

class_id_to_name = {
    0: "other",
    1: "lenin",
    2: "ataturk"
}

@exception_handler
def calculate_map(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> Dict[str, float]:
    """
    Calculate mAP (mean Average Precision) for the model on the given data loader
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to use for evaluation
        iou_thresholds: IoU thresholds for mAP calculation
        
    Returns:
        Dictionary with mAP metrics
    """
    model.eval()
    
    # Store all detections and ground truths
    all_detections = []
    all_ground_truths = []
    
    logger.info("Collecting detections and ground truths for mAP calculation...")
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move images to device
            images = list(image.to(device) for image in images)
            
            # Forward pass
            outputs = model(images)
            
            # Store detections and ground truths
            for output, target in zip(outputs, targets):
                # Get detections
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                
                # Get ground truths
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                
                # Add to lists
                all_detections.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                })
                
                all_ground_truths.append({
                    "boxes": gt_boxes,
                    "labels": gt_labels
                })
    
    # Calculate mAP
    logger.info("Calculating mAP...")
    
    # Initialize metrics
    metrics = {
        "mAP": 0.0,
        "mAP_50": 0.0,
        "mAP_75": 0.0,
        "mAP_small": 0.0,
        "mAP_medium": 0.0,
        "mAP_large": 0.0,
        "AR_max_1": 0.0,
        "AR_max_10": 0.0,
        "AR_max_100": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0
    }
    
    # Calculate mAP for each class
    for class_id in range(1, 3):  # Lenin and Ataturk (skip "other")
        class_name = class_id_to_name[class_id]
        
        # Calculate Average Precision for each IoU threshold
        aps = []
        
        for iou_threshold in iou_thresholds:
            ap = calculate_ap_for_class(
                all_detections=all_detections,
                all_ground_truths=all_ground_truths,
                class_id=class_id,
                iou_threshold=iou_threshold
            )
            aps.append(ap)
        
        # Calculate mAP for the class
        class_map = np.mean(aps)
        
        # Add to metrics
        metrics[f"mAP_{class_name}"] = class_map
        metrics[f"AP_50_{class_name}"] = aps[0]  # IoU 0.5
        metrics[f"AP_75_{class_name}"] = aps[5]  # IoU 0.75
        
        # Add to overall mAP
        metrics["mAP"] += class_map / 2  # Average over 2 classes
        metrics["mAP_50"] += aps[0] / 2
        metrics["mAP_75"] += aps[5] / 2
    
    logger.info(f"mAP: {metrics['mAP']:.4f}, mAP_50: {metrics['mAP_50']:.4f}, mAP_75: {metrics['mAP_75']:.4f}")
    
    return metrics

def calculate_ap_for_class(
    all_detections: List[Dict[str, np.ndarray]],
    all_ground_truths: List[Dict[str, np.ndarray]],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision for a specific class and IoU threshold
    
    Args:
        all_detections: List of detection dictionaries
        all_ground_truths: List of ground truth dictionaries
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for matching
        
    Returns:
        Average Precision value
    """
    # Collect all detections for this class
    all_class_detections = []
    
    for detections in all_detections:
        # Filter by class
        class_mask = detections["labels"] == class_id
        class_boxes = detections["boxes"][class_mask]
        class_scores = detections["scores"][class_mask]
        
        # Add to list
        for box, score in zip(class_boxes, class_scores):
            all_class_detections.append({
                "box": box,
                "score": score,
                "matched": False
            })
    
    # Sort detections by score
    all_class_detections = sorted(all_class_detections, key=lambda x: x["score"], reverse=True)
    
    # Count total ground truths for this class
    total_gt = 0
    
    for gt in all_ground_truths:
        class_mask = gt["labels"] == class_id
        total_gt += np.sum(class_mask)
    
    # If no ground truths, AP is 0
    if total_gt == 0:
        return 0.0
    
    # If no detections, AP is 0
    if len(all_class_detections) == 0:
        return 0.0
    
    # Calculate precision and recall
    tp = np.zeros(len(all_class_detections))
    fp = np.zeros(len(all_class_detections))
    
    # Iterate through detections
    for i, detection in enumerate(all_class_detections):
        # Get detection box
        det_box = detection["box"]
        
        # Check against all ground truths
        best_iou = 0.0
        best_gt_idx = -1
        best_gt_img_idx = -1
        
        for img_idx, gt in enumerate(all_ground_truths):
            # Filter by class
            class_mask = gt["labels"] == class_id
            gt_boxes = gt["boxes"][class_mask]
            
            # Check each ground truth box
            for gt_idx, gt_box in enumerate(gt_boxes):
                # Calculate IoU
                iou = calculate_iou(det_box, gt_box)
                
                # Update best match
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_img_idx = img_idx
        
        # If found a match
        if best_gt_idx >= 0:
            # Check if already matched
            already_matched = False
            
            for prev_det in all_class_detections[:i]:
                if prev_det["matched"] and prev_det.get("matched_img_idx") == best_gt_img_idx and prev_det.get("matched_gt_idx") == best_gt_idx:
                    already_matched = True
                    break
            
            if not already_matched:
                tp[i] = 1
                detection["matched"] = True
                detection["matched_img_idx"] = best_gt_img_idx
                detection["matched_gt_idx"] = best_gt_idx
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    # Calculate cumulative precision and recall
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    
    recall = cumsum_tp / total_gt
    precision = cumsum_tp / (cumsum_tp + cumsum_fp)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        
        ap += p / 11
    
    return ap

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes
    
    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate area of union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

@exception_handler
async def visualize_predictions(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    device: str = "cuda",
    score_threshold: float = 0.5
) -> None:
    """
    Visualize model predictions on an image
    
    Args:
        model: Model to use for prediction
        image_path: Path to the input image
        output_path: Path to save the output image
        device: Device to use for prediction
        score_threshold: Threshold for detection scores
    """
    # Set model to evaluation mode
    model.eval()
    
    # Load image
    image_np = await read_image_async(image_path)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    
    # Normalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Get predictions
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    # Get boxes, scores, and labels
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    
    # Filter by score
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create image for drawing
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    
    # Draw bounding boxes
    for box, score, label in zip(boxes, scores, labels):
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name
        class_name = class_id_to_name.get(label, f"unknown_{label}")
        
        # Choose color based on class
        if label == 1:  # Lenin
            color = (255, 0, 0)  # Red
        elif label == 2:  # Ataturk
            color = (0, 0, 255)  # Blue
        else:  # Other
            color = (0, 255, 0)  # Green
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        draw.text((x1, y1 - 10), text, fill=color)
    
    # Save image
    image_pil.save(output_path)
    
    logger.info(f"Visualization saved to {output_path}")

@exception_handler
async def visualize_batch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    output_dir: str,
    device: str = "cuda",
    num_samples: int = 10
) -> None:
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: Model to use for prediction
        data_loader: DataLoader for images
        output_dir: Directory to save output images
        device: Device to use for prediction
        num_samples: Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of images
    images, targets = next(iter(data_loader))
    
    # Limit to num_samples
    images = images[:num_samples]
    targets = targets[:num_samples]
    
    # Predict on images
    with torch.no_grad():
        predictions = model([image.to(device) for image in images])
    
    logger.info(f"Visualizing {len(images)} predictions...")
    
    # Loop through images and predictions
    for i, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
        # Convert image to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(image_np)
        
        # Get predicted boxes, scores, and labels
        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        
        # Get ground truth boxes and labels
        gt_boxes = target["boxes"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        
        # Filter predictions by score
        mask = pred_scores >= 0.5
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]
        
        # Draw predicted boxes
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, edgecolor="r", facecolor="none"
            )
            
            # Add rectangle to plot
            ax.add_patch(rect)
            
            # Add label
            class_name = class_id_to_name.get(label, f"unknown_{label}")
            ax.text(
                x1, y1 - 10, f"{class_name}: {score:.2f}",
                color="r", fontsize=8, backgroundcolor="white"
            )
        
        # Draw ground truth boxes
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, edgecolor="g", facecolor="none"
            )
            
            # Add rectangle to plot
            ax.add_patch(rect)
            
            # Add label
            class_name = class_id_to_name.get(label, f"unknown_{label}")
            ax.text(
                x1, y2 + 10, f"GT: {class_name}",
                color="g", fontsize=8, backgroundcolor="white"
            )
        
        # Set title
        ax.set_title(f"Image {i+1}: {target['filename']}")
        
        # Remove axis
        ax.axis("off")
        
        # Save figure
        output_path = os.path.join(output_dir, f"sample_{i+1}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    
    logger.info(f"Saved {len(images)} visualizations to {output_dir}")