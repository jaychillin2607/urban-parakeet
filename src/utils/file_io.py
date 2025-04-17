import os
import io
import csv
import json
import aiofiles
import asyncio
from typing import List, Dict, Any, Union, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import yaml

async def read_csv_async(filepath: str) -> pd.DataFrame:
    """
    Asynchronously read a CSV file and return a pandas DataFrame
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the CSV data
    """
    try:
        async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
            content = await f.read()
        
        # Process the CSV content
        return pd.read_csv(
            io.StringIO(content), 
            dtype={'filename': str}
        )
    except Exception as e:
        raise IOError(f"Error reading CSV file {filepath}: {str(e)}")

async def write_csv_async(
    filepath: str, 
    data: List[Dict[str, Any]], 
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ','
) -> None:
    """
    Asynchronously write a CSV file
    
    Args:
        filepath: Path to the output CSV file
        data: List of dictionaries to write
        fieldnames: Column names (if None, will use keys from first dict)
        delimiter: CSV delimiter character
    """
    try:
        if not fieldnames and data:
            fieldnames = list(data[0].keys())
            
        async with aiofiles.open(filepath, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames,
                delimiter=delimiter
            )
            await f.write(delimiter.join(fieldnames) + '\n')
            
            for row in data:
                line = delimiter.join(str(row.get(field, '')) for field in fieldnames)
                await f.write(line + '\n')
    except Exception as e:
        raise IOError(f"Error writing CSV file {filepath}: {str(e)}")

async def write_results_csv_async(
    filepath: str, 
    results: List[Dict[str, Union[str, int, float]]]
) -> None:
    """
    Write detection results to CSV in required format:
    {image name};{x1};{y1};{x2};{y2};{class}
    
    Args:
        filepath: Path to the output CSV file
        results: List of dictionaries with keys:
                 'filename', 'x1', 'y1', 'x2', 'y2', 'class'
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        async with aiofiles.open(filepath, mode='w', encoding='utf-8', newline='') as f:
            for result in results:
                line = f"{result['filename']};{result['x1']};{result['y1']};{result['x2']};{result['y2']};{result['class']}\n"
                await f.write(line)
    except Exception as e:
        raise IOError(f"Error writing results CSV file {filepath}: {str(e)}")

async def read_image_async(filepath: str) -> np.ndarray:
    """
    Asynchronously read an image file
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Image as numpy array
    """
    # For image reading, we use a thread executor to avoid blocking
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, lambda: np.array(Image.open(filepath).convert('RGB')))
    return img

async def save_image_async(filepath: str, image: np.ndarray) -> None:
    """
    Asynchronously save an image file
    
    Args:
        filepath: Path to save the image
        image: Numpy array containing the image data
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: Image.fromarray(image).save(filepath))

def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load YAML configuration file (synchronous since it's typically done once)
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        Dictionary with configuration values
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(filepath: str, data: Dict[str, Any]) -> None:
    """
    Save dictionary to YAML file (synchronous since it's typically done once)
    
    Args:
        filepath: Path to save the YAML file
        data: Dictionary to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)