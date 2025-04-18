#!/usr/bin/env python3
# debug_dataset.py
import os
import pandas as pd
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.append(".")

from src.config import config
from src.data.preprocessing import load_annotation_files
from src.utils.logging import get_logger
import asyncio

logger = get_logger("debug_dataset")

def draw_bounding_boxes(image_path, boxes, labels, colors=None, output_path=None):
    """Draw bounding boxes on an image"""
    if colors is None:
        colors = {
            "background": (200, 200, 200),
            "other": (0, 255, 0),
            "lenin": (255, 0, 0),
            "ataturk": (0, 0, 255)
        }
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw each box
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        color = colors.get(label, (255, 255, 0))
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        draw.text((x1, y1-15), label, fill=color)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
    else:
        return img

async def analyze_dataset():
    """Analyze the dataset and generate visualizations"""
    logger.info("Starting dataset analysis")
    
    # Load annotations
    df = await load_annotation_files()
    logger.info(f"Loaded {len(df)} annotations")
    
    # Basic statistics
    print(f"Total annotations: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    print(f"Image count: {df['filename'].nunique()}")
    
    # Box size statistics
    df['box_width'] = df['xmax'] - df['xmin']
    df['box_height'] = df['ymax'] - df['ymin']
    df['box_area'] = df['box_width'] * df['box_height']
    
    print("\nBounding box statistics:")
    print(f"Average width: {df['box_width'].mean():.1f} pixels")
    print(f"Average height: {df['box_height'].mean():.1f} pixels")
    print(f"Average area: {df['box_area'].mean():.1f} pixels")
    
    # Sample some images to visualize
    sample_count = 5
    sample_images = df['filename'].unique()
    if len(sample_images) > sample_count:
        sample_images = random.sample(list(sample_images), sample_count)
    
    os.makedirs("debug_output", exist_ok=True)
    
    # Visualize samples
    for filename in sample_images:
        # Get annotations for this image
        image_df = df[df['filename'] == filename]
        
        # Find image path
        image_path = None
        for dir_name in ["statues-lenin", "statues-ataturk", "statues-other"]:
            test_path = os.path.join(config.data.raw_data_dir, dir_name, filename)
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            logger.warning(f"Image {filename} not found")
            continue
        
        # Get boxes and labels
        boxes = []
        labels = []
        for _, row in image_df.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
        
        # Draw bounding boxes
        output_path = os.path.join("debug_output", filename)
        draw_bounding_boxes(image_path, boxes, labels, output_path=output_path)
        logger.info(f"Saved visualization for {filename}")
    
    # Generate plots for box sizes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['box_width'], bins=30)
    plt.title('Box Width Distribution')
    plt.xlabel('Width (pixels)')
    
    plt.subplot(1, 3, 2)
    plt.hist(df['box_height'], bins=30)
    plt.title('Box Height Distribution')
    plt.xlabel('Height (pixels)')
    
    plt.subplot(1, 3, 3)
    plt.hist(df['box_area'], bins=30)
    plt.title('Box Area Distribution')
    plt.xlabel('Area (pixels²)')
    
    plt.tight_layout()
    plt.savefig('debug_output/box_size_distribution.png')
    logger.info("Saved box size distribution plot")
    
    # Generate plot for class distribution
    plt.figure(figsize=(10, 6))
    class_counts = df['class'].value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('debug_output/class_distribution.png')
    logger.info("Saved class distribution plot")
    
    print("\nAnalysis complete. Visualizations saved to debug_output/")

if __name__ == "__main__":
    asyncio.run(analyze_dataset())