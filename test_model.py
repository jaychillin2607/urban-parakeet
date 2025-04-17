#!/usr/bin/env python3
import os
import unittest
import asyncio
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.config import config
from src.models.faster_rcnn import create_faster_rcnn_model
from src.data.dataset import StatueDataset
from src.utils.file_io import read_image_async
from src.inference.predict import predict_image, process_predictions

class TestStatueDetector(unittest.TestCase):
    """Test cases for statue detection model"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Check for GPU
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        cls.model = create_faster_rcnn_model(
            num_classes=3,
            backbone="resnet18",  # Smaller backbone for faster tests
            pretrained=False,     # No need for pretrained weights in tests
            box_score_thresh=0.5,
            box_nms_thresh=0.3,
            box_detections_per_img=10
        )
        cls.model = cls.model.to(cls.device)
        cls.model.eval()
        
        # Create test directory
        os.makedirs("test_outputs", exist_ok=True)
    
    def test_model_creation(self):
        """Test model creation"""
        # Create a fresh model
        model = create_faster_rcnn_model(
            num_classes=3,
            backbone="resnet18",
            pretrained=False
        )
        
        # Check model type
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check output layer dimensions
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        out_features = model.roi_heads.box_predictor.cls_score.out_features
        
        self.assertEqual(out_features, 3)  # 3 classes
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Create a test image
        image = torch.randn(3, 800, 800, device=self.device)
        images = [image]
        
        # Test forward pass for inference
        with torch.no_grad():
            predictions = self.model(images)
        
        # Check output format
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 1)
        
        # Check prediction keys
        pred = predictions[0]
        self.assertIn("boxes", pred)
        self.assertIn("labels", pred)
        self.assertIn("scores", pred)
        
        # Check data types
        self.assertEqual(pred["boxes"].device, torch.device(self.device))
        self.assertEqual(pred["labels"].device, torch.device(self.device))
        self.assertEqual(pred["scores"].device, torch.device(self.device))
    
    def test_model_training(self):
        """Test model training"""
        # Create a test image and target
        image = torch.randn(3, 800, 800, device=self.device)
        target = {
            "boxes": torch.tensor([[100, 100, 200, 200]], device=self.device),
            "labels": torch.tensor([1], device=self.device)  # Lenin class
        }
        
        # Switch model to training mode
        self.model.train()
        
        # Forward pass with target for training
        loss_dict = self.model([image], [target])
        
        # Check loss dict
        self.assertIn("loss_classifier", loss_dict)
        self.assertIn("loss_box_reg", loss_dict)
        self.assertIn("loss_objectness", loss_dict)
        self.assertIn("loss_rpn_box_reg", loss_dict)
        
        # Switch back to eval mode
        self.model.eval()
    
    async def async_test_prediction(self):
        """Test prediction on a real image (async)"""
        # Find test image
        test_image = None
        
        for class_dir in ["statues-lenin", "statues-ataturk", "statues-other"]:
            class_path = os.path.join(config.data.raw_data_dir, class_dir)
            if os.path.exists(class_path):
                images = os.listdir(class_path)
                if images:
                    test_image = os.path.join(class_path, images[0])
                    break
        
        if test_image is None:
            self.skipTest("No test images found")
            return
        
        # Run prediction
        prediction = await predict_image(
            model=self.model,
            image_path=test_image,
            device=self.device,
            score_threshold=0.5
        )
        
        # Check prediction keys
        self.assertIn("filename", prediction)
        self.assertIn("boxes", prediction)
        self.assertIn("scores", prediction)
        self.assertIn("labels", prediction)
        self.assertIn("image_shape", prediction)
        
        # Process prediction
        results = process_predictions([prediction])
        
        # Check results
        self.assertEqual(len(results), 1)
        result = results[0]
        
        self.assertIn("filename", result)
        self.assertIn("x1", result)
        self.assertIn("y1", result)
        self.assertIn("x2", result)
        self.assertIn("y2", result)
        self.assertIn("class", result)
        
        # Check result types
        self.assertIsInstance(result["x1"], (int, float))
        self.assertIsInstance(result["y1"], (int, float))
        self.assertIsInstance(result["x2"], (int, float))
        self.assertIsInstance(result["y2"], (int, float))
        self.assertIsInstance(result["class"], int)
    
    def test_prediction(self):
        """Wrapper for async test"""
        asyncio.run(self.async_test_prediction())
    
    def test_dataset(self):
        """Test dataset loading"""
        # Check if processed data exists
        if not os.path.exists(config.data.processed_data_dir):
            self.skipTest("Processed data not found. Run preprocessing first.")
            return
        
        # Check if annotations file exists
        annotation_file = config.data.train_annotations_file
        if not os.path.exists(annotation_file):
            self.skipTest(f"Annotation file {annotation_file} not found")
            return
        
        # Create dataset
        dataset = StatueDataset(
            annotation_file=annotation_file,
            transform=None,
            is_train=True
        )
        
        # Check dataset size
        self.assertGreater(len(dataset), 0)
        
        # Get a sample
        sample = dataset[0]
        
        # Check sample format
        self.assertIn("image", sample)
        self.assertIn("target", sample)
        
        # Check target format
        target = sample["target"]
        self.assertIn("boxes", target)
        self.assertIn("labels", target)
        self.assertIn("image_id", target)
        self.assertIn("filename", target)

if __name__ == "__main__":
    unittest.main()