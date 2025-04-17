import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class DataConfig:
    """Data configuration settings"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    results_dir: str = "data/results"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Class mapping - using default_factory to fix mutable default issue
    class_map: Dict[str, int] = field(default_factory=lambda: {
        "other": 0,
        "lenin": 1,
        "ataturk": 2
    })
    
    def __post_init__(self):
        # Initialize path fields after the basic fields are set
        # Original labels files
        self.statues_labels_file = os.path.join(self.raw_data_dir, "statues_labels.csv")
        self.statues_labels2_file = os.path.join(self.raw_data_dir, "statues_labels2.csv")
        
        # Statue class folders
        self.lenin_dir = os.path.join(self.raw_data_dir, "statues-lenin")
        self.ataturk_dir = os.path.join(self.raw_data_dir, "statues-ataturk")
        self.other_dir = os.path.join(self.raw_data_dir, "statues-other")
        
        # Processed data files
        self.train_annotations_file = os.path.join(self.processed_data_dir, "train_annotations.csv")
        self.val_annotations_file = os.path.join(self.processed_data_dir, "val_annotations.csv")
        self.test_annotations_file = os.path.join(self.processed_data_dir, "test_annotations.csv")
        
        # Results file
        self.results_file = os.path.join(self.results_dir, "results.csv")


@dataclass
class ModelConfig:
    """Model configuration settings"""
    backbone: str = "resnet50"
    pretrained: bool = True
    pretrained_weights: str = "COCO"
    num_classes: int = 3  # Background is handled automatically by Faster R-CNN
    checkpoint_dir: str = "models/checkpoints"
    
    # Model hyperparameters
    box_score_thresh: float = 0.5
    box_nms_thresh: float = 0.3
    box_detections_per_img: int = 100


@dataclass
class TrainConfig:
    """Training configuration settings"""
    device: str = "cuda"  # or "cpu" if GPU not available
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    epochs: int = 20
    log_interval: int = 10
    save_interval: int = 1
    early_stopping_patience: int = 5
    
    # Optimizer
    optimizer: str = "sgd"  # "adam", "sgd", etc.
    
    # Scheduler
    scheduler: str = "step"  # "step", "cosine", etc.
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.1
    
    # Logging
    log_dir: str = "logs"
    use_wandb: bool = True
    wandb_project: str = "statue-detection"
    
    # Transforms/Augmentations
    image_size: Tuple[int, int] = (800, 800)
    use_augmentations: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration settings"""
    checkpoint_path: str = ""  # Will be set during runtime
    batch_size: int = 1
    num_workers: int = 2
    device: str = "cuda"  # or "cpu" if GPU not available


@dataclass
class Config:
    """Main configuration class combining all config sections"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with values from dictionary"""
        for section, section_dict in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in section_dict.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)


# Create default configuration
config = Config()