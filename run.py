#!/usr/bin/env python3
import os
import argparse
import asyncio
import torch
from datetime import datetime
import logging

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import exception_handler, setup_global_exception_handler
from src.data.preprocessing import preprocess_dataset
from src.data.dataset import create_data_loaders
from src.models.faster_rcnn import create_faster_rcnn_model, save_model, load_model
from src.training.train import train_model
from src.inference.predict import predict_directory
from src.inference.visualization import create_visualization_report

logger = get_logger("main")

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Statue Detection and Classification")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--epochs", type=int, default=config.train.epochs, 
                            help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=config.train.batch_size,
                            help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=config.train.learning_rate,
                            help="Learning rate")
    train_parser.add_argument("--device", type=str, default=config.train.device,
                            help="Device to use (cuda or cpu)")
    train_parser.add_argument("--backbone", type=str, default=config.model.backbone,
                            help="Backbone model (resnet18, resnet34, resnet50, resnet101)")
    train_parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to checkpoint for resuming training")
    train_parser.add_argument("--no-wandb", action="store_true", 
                            help="Disable wandb logging")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on images")
    inference_parser.add_argument("--input", type=str, required=True,
                                help="Directory with input images")
    inference_parser.add_argument("--output", type=str, default=config.data.results_file,
                                help="Path to output CSV file")
    inference_parser.add_argument("--checkpoint", type=str, required=True,
                                help="Path to model checkpoint")
    inference_parser.add_argument("--device", type=str, default=config.inference.device,
                                help="Device to use (cuda or cpu)")
    inference_parser.add_argument("--threshold", type=float, default=config.model.box_score_thresh,
                                help="Score threshold for detections")
    inference_parser.add_argument("--backbone", type=str, default=config.model.backbone,
                            help="Backbone model (resnet18, resnet34, resnet50, resnet101)")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize detection results")
    visualize_parser.add_argument("--results", type=str, default=config.data.results_file,
                                help="Path to results CSV file")
    visualize_parser.add_argument("--input", type=str, default=config.data.raw_data_dir,
                                help="Directory with input images")
    visualize_parser.add_argument("--output", type=str, default="data/visualizations",
                                help="Directory to save visualizations")
    visualize_parser.add_argument("--samples", type=int, default=20,
                                help="Number of sample images to visualize")
    
    return parser.parse_args()

@exception_handler
async def run_preprocess():
    """
    Run data preprocessing
    """
    logger.info("Starting data preprocessing...")
    await preprocess_dataset()
    logger.info("Data preprocessing completed")

@exception_handler
async def run_train(args):
    """
    Run model training
    
    Args:
        args: Command line arguments
    """
    # Update config with command line args
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.model.backbone = args.backbone
    config.train.use_wandb = not args.no_wandb
    
    # Check if GPU is available
    if config.train.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU instead")
        config.train.device = "cpu"
    
    # Preprocess data if needed
    if not os.path.exists(config.data.train_annotations_file):
        logger.info("Preprocessing data first...")
        await preprocess_dataset()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Create or load model
    if args.checkpoint is not None:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model, optimizer, metadata = load_model(args.checkpoint, config.train.device)
        start_epoch = metadata.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        logger.info("Creating new model...")
        model = create_faster_rcnn_model(
            num_classes=config.model.num_classes,
            backbone=config.model.backbone,
            pretrained=config.model.pretrained,
            box_score_thresh=config.model.box_score_thresh,
            box_nms_thresh=config.model.box_nms_thresh,
            box_detections_per_img=config.model.box_detections_per_img
        )
        model = model.to(config.train.device)
        start_epoch = 0
    
    # Train model
    logger.info("Starting model training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.train.device,
        start_epoch=start_epoch,
        num_epochs=config.train.epochs,
        log_interval=config.train.log_interval,
        save_interval=config.train.save_interval,
        early_stopping_patience=config.train.early_stopping_patience,
        checkpoint_dir=config.model.checkpoint_dir
    )
    
    logger.info("Model training completed")

@exception_handler
async def run_inference(args):
    """
    Run inference on images
    
    Args:
        args: Command line arguments
    """
    # Update config with command line args
    config.inference.device = args.device
    config.data.results_file = args.output
    config.model.box_score_thresh = args.threshold
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        raise ValueError(f"Input directory {args.input} does not exist")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model, _, _ = load_model(args.checkpoint, config.inference.device,backbone=args.backbone )
    
    # Create output directory
    os.makedirs(os.path.dirname(config.data.results_file), exist_ok=True)
    
    # Run inference
    logger.info(f"Running inference on images in {args.input}...")
    await predict_directory(
        model=model,
        input_dir=args.input,
        output_file=config.data.results_file,
        device=config.inference.device,
        score_threshold=config.model.box_score_thresh
    )
    
    logger.info(f"Inference completed. Results saved to {config.data.results_file}")

@exception_handler
async def run_visualize(args):
    """
    Visualize detection results
    
    Args:
        args: Command line arguments
    """
    # Check if results file exists
    if not os.path.exists(args.results):
        raise ValueError(f"Results file {args.results} does not exist")
    
    # Create visualization report
    logger.info(f"Creating visualization report...")
    await create_visualization_report(
        results_csv=args.results,
        input_dir=args.input,
        output_dir=args.output,
        sample_count=args.samples
    )
    
    logger.info(f"Visualization completed. Results saved to {args.output}")

@exception_handler
async def main():
    """
    Main entry point
    """
    # Set up global exception handler
    setup_global_exception_handler()
    
    # Parse arguments
    args = parse_args()
    
    # Run appropriate mode
    if args.mode == "preprocess":
        await run_preprocess()
    elif args.mode == "train":
        await run_train(args)
    elif args.mode == "inference":
        await run_inference(args)
    elif args.mode == "visualize":
        await run_visualize(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        print("Please specify a mode: preprocess, train, inference, or visualize")

if __name__ == "__main__":
    asyncio.run(main())