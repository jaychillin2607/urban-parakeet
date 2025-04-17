import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import wandb
from tqdm import tqdm

from src.config import config
from src.utils.logging import get_logger
from src.utils.exceptions import TrainingError, exception_handler
from src.models.faster_rcnn import create_faster_rcnn_model, save_model, load_model

logger = get_logger("training")

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 5, min_delta: float = 0):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait after validation loss stops improving
            min_delta: Minimum change in validation loss to be considered an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered")
                self.early_stop = True
                
        return self.early_stop

def get_optimizer(
    model: nn.Module, 
    optimizer_name: str = "sgd",
    lr: float = 0.001,
    weight_decay: float = 0.0005,
    momentum: float = 0.9
) -> torch.optim.Optimizer:
    """
    Get optimizer for model training
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer (sgd, adam, etc.)
        lr: Learning rate
        weight_decay: Weight decay factor
        momentum: Momentum factor (for SGD)
        
    Returns:
        Configured optimizer
    """
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise TrainingError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "step",
    step_size: int = 5,
    gamma: float = 0.1,
    epochs: int = 20
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler (step, cosine, etc.)
        step_size: Step size for StepLR
        gamma: Gamma factor for StepLR
        epochs: Total number of epochs
        
    Returns:
        Configured scheduler
    """
    if scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    elif scheduler_name.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=gamma,
            patience=step_size // 2,
            verbose=True
        )
    elif scheduler_name.lower() == "none":
        return None
    else:
        raise TrainingError(f"Unsupported scheduler: {scheduler_name}")

@exception_handler
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: str,
    epoch: int,
    log_interval: int = 10
) -> float:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer for training
        data_loader: DataLoader for training data
        device: Device to use for training
        epoch: Current epoch number
        log_interval: Interval for logging
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    # Create progress bar - Fix: ensure leave=True for the main bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", leave=True, position=0)
    
    # Track time
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass (Faster R-CNN returns losses directly)
        loss_dict = model(images, targets)
        
        # Calculate total loss - handle list of dictionaries
        if isinstance(loss_dict, list) and all(isinstance(d, dict) for d in loss_dict):
            # List of dictionaries - sum all values from all dictionaries
            losses = sum(loss for d in loss_dict for loss in d.values())
        elif isinstance(loss_dict, dict):
            # Single dictionary - sum the values
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(loss_dict, list):
            # Simple list - sum the elements
            losses = sum(loss_dict)
        else:
            # If it's already a scalar tensor
            losses = loss_dict
        
        # Backward pass and optimize
        losses.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += losses.detach().item()
        
        # Update progress bar - Fix: use a float format for the loss
        pbar.set_postfix({"loss": f"{losses.detach().item():.4f}"})
        
        # Log to wandb
        if config.train.use_wandb and batch_idx % log_interval == 0:
            wandb.log({
                "batch_loss": losses.item(),
                "batch_rpn_box_loss": loss_dict["loss_rpn_box_reg"].item(),
                "batch_rpn_cls_loss": loss_dict["loss_objectness"].item(),
                "batch_box_loss": loss_dict["loss_box_reg"].item(),
                "batch_cls_loss": loss_dict["loss_classifier"].item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch": batch_idx + epoch * len(data_loader)
            })
    
    # Calculate average loss
    avg_loss = epoch_loss / len(data_loader)
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    
    logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    return avg_loss

@exception_handler
def validate(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    epoch: int
) -> float:
    """
    Validate model - inference-only mode without loss calculation
    
    Args:
        model: Model to validate
        data_loader: DataLoader for validation data
        device: Device to use for validation
        epoch: Current epoch number
        
    Returns:
        Average confidence score (as a proxy for validation metric)
    """
    model.eval()
    total_score = 0.0
    total_detections = 0
    processed_batches = 0
    
    # Class-wise metrics
    class_detections = {0: 0, 1: 0, 2: 0}  # Other, Lenin, Ataturk
    
    # Create progress bar
    pbar = tqdm(data_loader, desc=f"Validation", leave=True, position=0)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Move images to device - we don't need targets for inference
                images = list(image.to(device) for image in images)
                
                # Run model in inference mode (without targets)
                predictions = model(images)
                
                # Process predictions
                batch_score = 0.0
                batch_detections = 0
                
                for pred in predictions:
                    # Get scores and labels
                    scores = pred["scores"].detach().cpu().numpy()
                    labels = pred["labels"].detach().cpu().numpy()
                    
                    # Apply score threshold
                    mask = scores >= config.model.box_score_thresh
                    scores = scores[mask]
                    labels = labels[mask]
                    
                    # Count detections
                    num_detections = len(scores)
                    batch_detections += num_detections
                    
                    # Sum confidence scores
                    if num_detections > 0:
                        batch_score += scores.sum()
                        
                        # Update class-wise metrics
                        for label in labels:
                            if label in class_detections:
                                class_detections[label] += 1
                
                # Update metrics
                total_score += batch_score
                total_detections += batch_detections
                processed_batches += 1
                
                # Update progress bar
                avg_confidence = batch_score / max(batch_detections, 1)
                pbar.set_postfix({
                    "detections": batch_detections, 
                    "avg_conf": f"{avg_confidence:.4f}"
                })
                
            except Exception as e:
                logger.warning(f"Error in validation batch {batch_idx}: {str(e)}")
                # Continue with next batch
                continue
    
    # Calculate overall metrics
    avg_confidence = total_score / max(total_detections, 1)
    avg_detections = total_detections / max(processed_batches, 1)
    
    # Log validation results
    logger.info(f"Epoch {epoch+1} - Validation Summary:")
    logger.info(f"  Processed batches: {processed_batches}/{len(data_loader)}")
    logger.info(f"  Total detections: {total_detections} (avg: {avg_detections:.2f} per batch)")
    logger.info(f"  Average confidence: {avg_confidence:.4f}")
    logger.info(f"  Class distribution: Other: {class_detections[0]}, Lenin: {class_detections[1]}, Ataturk: {class_detections[2]}")
    
    # Log to wandb
    if config.train.use_wandb:
        wandb.log({
            "val_avg_confidence": avg_confidence,
            "val_total_detections": total_detections,
            "val_avg_detections": avg_detections,
            "val_other_detections": class_detections[0],
            "val_lenin_detections": class_detections[1],
            "val_ataturk_detections": class_detections[2],
            "epoch": epoch
        })
    
    return avg_confidence

@exception_handler
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    start_epoch: int = 0,
    num_epochs: int = 20,
    log_interval: int = 10,
    save_interval: int = 1,
    early_stopping_patience: int = 5,
    checkpoint_dir: str = "models/checkpoints"
) -> nn.Module:
    """
    Train model for multiple epochs
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        start_epoch: Starting epoch (for resuming training)
        num_epochs: Number of epochs to train
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Trained model
    """
    # Initialize wandb if enabled
    if config.train.use_wandb:
        wandb.init(
            project=config.train.wandb_project,
            config={
                "learning_rate": config.train.learning_rate,
                "epochs": config.train.epochs,
                "batch_size": config.train.batch_size,
                "optimizer": config.train.optimizer,
                "scheduler": config.train.scheduler,
                "model": config.model.backbone,
                "image_size": config.train.image_size
            }
        )
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config.train.optimizer,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        momentum=config.train.momentum
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config.train.scheduler,
        step_size=config.train.scheduler_step_size,
        gamma=config.train.scheduler_gamma,
        epochs=config.train.epochs
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Track best model (using confidence score instead of loss)
    best_val_metric = 0.0  # Higher confidence is better
    
    # Train for multiple epochs
    for epoch in range(start_epoch, num_epochs):
        # Train one epoch
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            log_interval=log_interval
        )
        
        # Validate
        val_metric = validate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # For ReduceLROnPlateau, we want to minimize loss
                scheduler.step(-val_metric)  # Negate confidence to make it act like loss
            else:
                scheduler.step()
        
        # Log to wandb
        if config.train.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_confidence": val_metric,
                "epoch": epoch
            })
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"statue_detector_epoch_{epoch+1}.pth"
            )
            save_model(
                model=model,
                filepath=checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                accuracy=val_metric
            )
        
        # Save best model (using confidence as metric)
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_checkpoint_path = os.path.join(
                checkpoint_dir,
                "statue_detector_best.pth"
            )
            save_model(
                model=model,
                filepath=best_checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                accuracy=val_metric
            )
            logger.info(f"Saved best model with validation confidence: {val_metric:.4f}")
        
        # Check early stopping
        # For confidence, we need to negate it since EarlyStopping expects decreasing metric
        if early_stopping(-val_metric):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Finish wandb
    if config.train.use_wandb:
        wandb.finish()
    
    return model