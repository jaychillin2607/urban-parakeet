import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import asyncio
from typing import Dict, List, Tuple, Optional, Union, Any

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import exception_handler, async_exception_handler
from src.utils.file_io import read_image_async, save_image_async

logger = get_logger("visualization")

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),    # Other - Green
    1: (255, 0, 0),    # Lenin - Red
    2: (0, 0, 255)     # Ataturk - Blue
}

# Class names
CLASS_NAMES = {
    0: "Other",
    1: "Lenin",
    2: "Ataturk"
}

@async_exception_handler
async def visualize_prediction(
    image_path: str,
    output_path: str,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.5
) -> None:
    """
    Visualize predictions on an image and save the result
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        boxes: Bounding boxes (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        score_threshold: Threshold for confidence scores
    """
    try:
        # Load image
        image = await read_image_async(image_path)
        
        # Create PIL image for drawing
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        
        # Filter by score threshold
        mask = scores >= score_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        # Draw bounding boxes
        for box, label, score in zip(boxes, labels, scores):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Get color and class name
            color = CLASS_COLORS.get(label, (255, 255, 255))
            class_name = CLASS_NAMES.get(label, f"Class {label}")
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text = f"{class_name}: {score:.2f}"
            text_size = draw.textsize(text)
            draw.rectangle([x1, y1 - text_size[1] - 4, x1 + text_size[0] + 4, y1], fill=color)
            
            # Draw text
            draw.text((x1 + 2, y1 - text_size[1] - 2), text, fill=(255, 255, 255))
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_pil.save(output_path)
        
    except Exception as e:
        logger.error(f"Error visualizing prediction for {image_path}: {str(e)}")

@async_exception_handler
async def visualize_results(
    results_csv: str,
    input_dir: str,
    output_dir: str,
    max_images: int = 100
) -> None:
    """
    Visualize detection results from the results CSV file
    
    Args:
        results_csv: Path to results CSV file
        input_dir: Directory with input images
        output_dir: Directory to save output images
        max_images: Maximum number of images to visualize
    """
    try:
        # Load results
        import pandas as pd
        results = pd.read_csv(results_csv, sep=';', header=None)
        results.columns = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Limit number of images
        if len(results) > max_images:
            logger.info(f"Limiting visualization to {max_images} images")
            results = results.sample(max_images)
        
        # Process results
        tasks = []
        
        for _, row in tqdm(results.iterrows(), total=len(results), desc="Visualizing"):
            # Get data
            filename = row['filename']
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            class_id = row['class']
            
            # Skip dummy entries
            if x1 == 0 and y1 == 0 and x2 == 1 and y2 == 1 and class_id == 0:
                continue
            
            # Find image path
            image_path = None
            
            # Check in Lenin directory
            lenin_path = os.path.join(input_dir, "statues-lenin", filename)
            if os.path.exists(lenin_path):
                image_path = lenin_path
            
            # Check in Ataturk directory
            if image_path is None:
                ataturk_path = os.path.join(input_dir, "statues-ataturk", filename)
                if os.path.exists(ataturk_path):
                    image_path = ataturk_path
            
            # Check in Other directory
            if image_path is None:
                other_path = os.path.join(input_dir, "statues-other", filename)
                if os.path.exists(other_path):
                    image_path = other_path
            
            # Skip if image not found
            if image_path is None:
                logger.warning(f"Image {filename} not found")
                continue
            
            # Create output path
            output_path = os.path.join(output_dir, filename)
            
            # Create numpy arrays
            boxes = np.array([[x1, y1, x2, y2]])
            labels = np.array([class_id])
            scores = np.array([1.0])  # Assume perfect score for results
            
            # Add task
            task = visualize_prediction(
                image_path=image_path,
                output_path=output_path,
                boxes=boxes,
                labels=labels,
                scores=scores
            )
            tasks.append(task)
        
        # Run tasks
        if tasks:
            logger.info(f"Visualizing {len(tasks)} images...")
            await asyncio.gather(*tasks)
            logger.info(f"Visualization complete. Results saved to {output_dir}")
        else:
            logger.warning("No visualization tasks created")
            
    except Exception as e:
        logger.error(f"Error visualizing results: {str(e)}")

@exception_handler
def create_class_distribution_chart(results_csv: str, output_path: str) -> None:
    """
    Create a chart showing the distribution of classes in the results
    
    Args:
        results_csv: Path to results CSV file
        output_path: Path to save the chart
    """
    try:
        # Load results
        import pandas as pd
        results = pd.read_csv(results_csv, sep=';', header=None)
        results.columns = ['filename', 'x1', 'y1', 'x2', 'y2', 'class']
        
        # Count classes
        class_counts = results['class'].value_counts().sort_index()
        
        # Create labels
        labels = [CLASS_NAMES.get(class_id, f"Class {class_id}") for class_id in class_counts.index]
        
        # Create colors
        colors = [CLASS_COLORS.get(class_id, (255, 255, 255)) for class_id in class_counts.index]
        rgb_colors = [f"rgb({r}, {g}, {b})" for b, g, r in colors]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        plt.bar(labels, class_counts.values, color=rgb_colors)
        
        # Add labels
        plt.title("Distribution of Statue Classes")
        plt.xlabel("Class")
        plt.ylabel("Count")
        
        # Add value labels
        for i, v in enumerate(class_counts.values):
            plt.text(i, v + 1, str(v), ha='center')
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Class distribution chart saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating class distribution chart: {str(e)}")

@exception_handler
async def create_visualization_report(
    results_csv: str,
    input_dir: str,
    output_dir: str,
    sample_count: int = 20
) -> None:
    """
    Create a comprehensive visualization report
    
    Args:
        results_csv: Path to results CSV file
        input_dir: Directory with input images
        output_dir: Directory to save output
        sample_count: Number of sample images to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class distribution chart
    chart_path = os.path.join(output_dir, "class_distribution.png")
    create_class_distribution_chart(results_csv, chart_path)
    
    # Visualize sample results
    samples_dir = os.path.join(output_dir, "samples")
    await visualize_results(
        results_csv=results_csv,
        input_dir=input_dir,
        output_dir=samples_dir,
        max_images=sample_count
    )
    
    logger.info(f"Visualization report created in {output_dir}")