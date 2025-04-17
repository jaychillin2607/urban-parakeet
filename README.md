# Statue Detection and Classification

This project implements a deep learning model to detect and classify statue heads in images. The model can identify if a statue depicts Vladimir Lenin, Mustafa Kemal Ataturk, or someone else.

## Project Overview

The system uses Faster R-CNN with a ResNet50 backbone to perform both detection (locating the head in the image) and classification (determining who the statue depicts). 

Key features:
- Detection of statue heads in images
- Classification into three categories: Lenin, Ataturk, or Other
- Asynchronous data processing for efficiency
- Comprehensive logging and error handling
- Visualization tools for results analysis
- Command-line interface for all operations

## Installation

1. Clone the repository:
```bash
git clone https://git.toptal.com/screening-ops/Nidhi-james-gopalkrishnan
cd Nidhi-james-gopalkrishnan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
statue-detection/
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Processed training data
│   └── results/             # Model predictions
├── models/
│   └── checkpoints/         # Saved model weights
├── src/
│   ├── data/                # Data handling modules
│   ├── models/              # Model definitions
│   ├── training/            # Training functions
│   ├── inference/           # Inference functions
│   └── utils/               # Utility functions
├── logs/                    # Log files
├── run.py                   # Main script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Usage

### Data Preprocessing

Before training, you need to preprocess the data:

```bash
python run.py preprocess
```

This will:
- Parse the annotation files
- Split the data into train, validation, and test sets
- Save the processed data to `data/processed/`

### Training

To train the model:

```bash
python run.py train --epochs 20 --batch-size 8 --lr 0.001 --device cuda
```

Additional options:
- `--backbone`: Choose backbone architecture (default: resnet50)
- `--checkpoint`: Resume training from a checkpoint
- `--no-wandb`: Disable Weights & Biases logging

### Inference

To run inference on a directory of images:

```bash
python run.py inference --input /path/to/images --output data/results/results.csv --checkpoint models/checkpoints/statue_detector_best.pth
```

This will:
- Load the model from the specified checkpoint
- Process all images in the input directory
- Generate a `results.csv` file in the format: `{image name};{x1};{y1};{x2};{y2};{class}`

### Visualization

To visualize the detection results:

```bash
python run.py visualize --results data/results/results.csv --input data/raw --output data/visualizations
```

This will:
- Create a class distribution chart
- Generate visualizations of sample detections
- Save the outputs to the specified directory

## Model Details

- Architecture: Faster R-CNN
- Backbone: ResNet50 with Feature Pyramid Network (FPN)
- Pretrained: COCO weights
- Classification: 3 classes (Lenin=1, Ataturk=2, Other=0)
- Detection: Head bounding boxes

## Data Handling Rules

The model follows these rules when processing images:
- If no head is found, it outputs `{image name};0;0;1;1;0`
- If there are multiple heads, it prioritizes:
  1. A single Lenin statue (if detected)
  2. A single Ataturk statue (if detected)
  3. The largest detected head (if multiple or only "other" statues)

## Performance

The model's performance can be evaluated using:
- mAP (mean Average Precision) at different IoU thresholds
- Class-specific precision and recall
- Visual inspection of the detection results

## Acknowledgments

This project was developed as part of the Toptal screening process.