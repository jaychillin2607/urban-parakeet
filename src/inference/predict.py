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
    score_threshold: float = 0.5
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
        
        # Get boxes, scores, and labels
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        
        # Filter by score
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
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
        
        other_class = 1
        lenin_class = 2    # Lenin's internal class ID
        ataturk_class = 3 
        # Check if Lenin or Ataturk is detected
        lenin_boxes = []
        ataturk_boxes = []
        other_boxes = []
        
        for box, score, label in zip(boxes, scores, labels):
            if label == lenin_class:  # Lenin
                lenin_boxes.append((box, score))
            elif label == ataturk_class:  # Ataturk
                ataturk_boxes.append((box, score))
            elif label == other_class:  # Other
                other_boxes.append((box, score))
        
        # Follow the rule:
        # If there are multiple heads, print only the coordinates of Lenin's or Ataturk's head;
        # if there are none or more than one, print the coordinates of the largest bounding box.
        
        selected_box = None
        selected_label = 0
        
        # First priority: Lenin
        if len(lenin_boxes) == 1:
            selected_box, _ = lenin_boxes[0]
            selected_label = lenin_class
        # Second priority: Ataturk
        elif len(ataturk_boxes) == 1:
            selected_box, _ = ataturk_boxes[0]
            selected_label = ataturk_class
        # If multiple Lenin or Ataturk or only other, select largest
        else:
            all_boxes = lenin_boxes + ataturk_boxes + other_boxes
            if all_boxes:
                # Find largest box by area
                largest_area = 0
                largest_idx = 0
                
                for i, (box, _) in enumerate(all_boxes):
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > largest_area:
                        largest_area = area
                        largest_idx = i
                
                selected_box, _ = all_boxes[largest_idx]
                
                # Determine label
                if largest_idx < len(lenin_boxes):
                    selected_label = lenin_class
                elif largest_idx < len(lenin_boxes) + len(ataturk_boxes):
                    selected_label = ataturk_class
                else:
                    selected_label = other_class
        
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