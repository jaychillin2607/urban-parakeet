import os
import torch
import numpy as np
from PIL import Image
import glob
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import asyncio

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import InferenceError, exception_handler, async_exception_handler
from src.utils.file_io import read_image_async, write_results_csv_async
from src.models.faster_rcnn import load_model

logger = get_logger("inference")

@exception_handler
def find_image_files(directory: str) -> List[str]:
    """
    Find all image files in a directory
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of image file paths
    """
    # Define supported image extensions
    extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    
    # Find all image files
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    
    if not image_files:
        logger.warning(f"No image files found in {directory}")
    
    return image_files

@async_exception_handler
async def predict_image(
    model: torch.nn.Module,
    image_path: str,
    device: str,
    score_threshold: float = 0.01  # Lower threshold to capture more detections
) -> Dict[str, Any]:
    """
    Make predictions on a single image
    
    Args:
        model: Model for prediction
        image_path: Path to the image
        device: Device to use for prediction
        score_threshold: Threshold for detection scores
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get filename
        filename = os.path.basename(image_path)
        
        # Load image
        image_np = await read_image_async(image_path)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model([image_tensor])[0]
        
        # Log raw prediction for debugging
        logger.debug(f"Raw prediction for {filename}:")
        logger.debug(f"  Boxes: {prediction['boxes'].shape}")
        logger.debug(f"  Scores: {prediction['scores']}")
        logger.debug(f"  Labels: {prediction['labels']}")
        
        # Get boxes, scores, and labels
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        
        # Filter by score
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        logger.info(f"Found {len(boxes)} detections with score >= {score_threshold} for {filename}")
        
        return {
            "filename": filename,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "image_shape": image_np.shape[:2]  # (height, width)
        }
        
    except Exception as e:
        logger.error(f"Error predicting image {image_path}: {str(e)}")
        
        # Return empty predictions
        return {
            "filename": os.path.basename(image_path),
            "boxes": np.array([]),
            "scores": np.array([]),
            "labels": np.array([]),
            "image_shape": (0, 0)
        }

@exception_handler
def process_predictions(
    predictions: List[Dict[str, Any]]
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Process raw predictions into the format required for results.csv
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        List of formatted results for writing to CSV
    """
    results = []
    
    # Log overall detection statistics
    total_images = len(predictions)
    images_with_detections = sum(1 for pred in predictions if len(pred["boxes"]) > 0)
    logger.info(f"Processing {total_images} images, {images_with_detections} with detections")
    
    for pred in predictions:
        filename = pred["filename"]
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        
        # If no heads found
        if len(boxes) == 0:
            result = {
                "filename": filename,
                "x1": 0,
                "y1": 0,
                "x2": 1,
                "y2": 1,
                "class": 0  # other
            }
            results.append(result)
            continue
        
        # Our internal class IDs based on config.data.class_map
        # {
        #   "background": 0,  # Add explicit background class
        #   "other": 1,       # Shift all classes up by 1
        #   "lenin": 2,
        #   "ataturk": 3
        # }
        
        # Output format: class is 0 for other, 1 for Lenin, 2 for Ataturk
        other_internal_id = 1    # Maps to 0 in output
        lenin_internal_id = 2    # Maps to 1 in output
        ataturk_internal_id = 3  # Maps to 2 in output
        
        # Convert internal class IDs to output format
        output_class_map = {
            other_internal_id: 0,    # Other -> 0
            lenin_internal_id: 1,    # Lenin -> 1
            ataturk_internal_id: 2   # Ataturk -> 2
        }
        
        # Check if Lenin or Ataturk is detected
        lenin_boxes = []
        ataturk_boxes = []
        other_boxes = []
        
        for box, score, label in zip(boxes, scores, labels):
            if label == lenin_internal_id:
                lenin_boxes.append((box, score))
            elif label == ataturk_internal_id:
                ataturk_boxes.append((box, score))
            elif label == other_internal_id:
                other_boxes.append((box, score))
        
        # Debug info
        logger.debug(f"{filename}: Found {len(lenin_boxes)} Lenin, {len(ataturk_boxes)} Ataturk, {len(other_boxes)} Other statues")
        
        # Follow the rule:
        # If there are multiple heads, print only coordinates of Lenin's or Ataturk's head;
        # if there are none or more than one, print coordinates of the largest bounding box.
        
        selected_box = None
        selected_label = 0  # Default to "other" (0 in output format)
        
        # First priority: Lenin
        if len(lenin_boxes) == 1:
            selected_box, _ = lenin_boxes[0]
            selected_label = output_class_map[lenin_internal_id]  # 1
            logger.debug(f"{filename}: Selected single Lenin statue")
        # Second priority: Ataturk
        elif len(ataturk_boxes) == 1:
            selected_box, _ = ataturk_boxes[0]
            selected_label = output_class_map[ataturk_internal_id]  # 2
            logger.debug(f"{filename}: Selected single Ataturk statue")
        # If multiple Lenin or Ataturk or only other, select largest
        else:
            all_boxes = []
            # Add Lenin boxes with their class
            for box, score in lenin_boxes:
                all_boxes.append((box, score, lenin_internal_id))
            # Add Ataturk boxes with their class
            for box, score in ataturk_boxes:
                all_boxes.append((box, score, ataturk_internal_id))
            # Add Other boxes with their class
            for box, score in other_boxes:
                all_boxes.append((box, score, other_internal_id))
            
            if all_boxes:
                # Find largest box by area
                largest_area = 0
                largest_idx = 0
                
                for i, (box, _, _) in enumerate(all_boxes):
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > largest_area:
                        largest_area = area
                        largest_idx = i
                
                selected_box, _, selected_class = all_boxes[largest_idx]
                selected_label = output_class_map[selected_class]
                logger.debug(f"{filename}: Selected largest box with class {selected_label}")
        
        # If still no box found (shouldn't happen), use default
        if selected_box is None:
            result = {
                "filename": filename,
                "x1": 0,
                "y1": 0,
                "x2": 1,
                "y2": 1,
                "class": 0
            }
        else:
            x1, y1, x2, y2 = map(int, selected_box)
            
            result = {
                "filename": filename,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class": selected_label
            }
        
        results.append(result)
    
    return results

@async_exception_handler
async def predict_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_file: str,
    device: str = "cuda",
    score_threshold: float = 0.5,
    batch_size: int = 8
) -> None:
    """
    Run prediction on all images in a directory and save results
    
    Args:
        model: Model for prediction
        input_dir: Directory with input images
        output_file: Path to output CSV file
        device: Device to use for prediction
        score_threshold: Threshold for detection scores
        batch_size: Number of images to process in parallel
    """
    # Find all image files
    image_files = find_image_files(input_dir)
    logger.info(f"Found {len(image_files)} image files in {input_dir}")
    
    # Process images in batches for parallel processing
    predictions = []
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
        # Get batch
        batch_files = image_files[i:i + batch_size]
        
        # Process batch in parallel
        tasks = [predict_image(model, image_path, device, score_threshold) for image_path in batch_files]
        batch_predictions = await asyncio.gather(*tasks)
        
        # Add to list
        predictions.extend(batch_predictions)
    
    # Process predictions
    results = process_predictions(predictions)
    
    # Write results to CSV
    await write_results_csv_async(output_file, results)
    
    logger.info(f"Predictions saved to {output_file}")

@exception_handler
async def main(
    checkpoint_path: str, 
    input_dir: str,
    output_file: str,
    device: str = "cuda"
) -> None:
    """
    Main function to run inference
    
    Args:
        checkpoint_path: Path to the model checkpoint
        input_dir: Directory with input images
        output_file: Path to output CSV file
        device: Device to use for inference
    """
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead")
        device = "cpu"
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model, _, _ = load_model(checkpoint_path, device)
    
    # Run prediction
    await predict_directory(
        model=model,
        input_dir=input_dir,
        output_file=output_file,
        device=device,
        score_threshold=config.model.box_score_thresh
    )

if __name__ == "__main__":
    # Example usage
    asyncio.run(main(
        checkpoint_path="models/checkpoints/statue_detector_best.pth",
        input_dir="data/raw/statues-test",
        output_file="data/results/results.csv",
        device="cuda"
    ))