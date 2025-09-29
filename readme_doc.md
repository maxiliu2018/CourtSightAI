# 🏀 Basketball Player Detection & Tracking System

Complete end-to-end pipeline for basketball player detection, jersey number recognition, and tracking using RT-DETR and SAM2.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Module Documentation](#module-documentation)
6. [Training Pipeline](#training-pipeline)
7. [Inference & Deployment](#inference--deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

This system provides a complete solution for:
- **Data Preparation**: Download, verify, and split COCO format datasets
- **Model Training**: Train RT-DETR models for player and jersey number detection
- **Video Tracking**: Track players across video frames with SAM2 integration
- **Real-time Inference**: Deploy models for real-time detection on images, videos, and webcams
- **Evaluation**: Comprehensive model evaluation with COCO metrics

### Key Features

- ✅ End-to-end training pipeline
- ✅ COCO format dataset handling
- ✅ RT-DETR model training with augmentation
- ✅ Player tracking with IoU-based tracking
- ✅ Real-time inference capabilities
- ✅ Model export (ONNX, TorchScript)
- ✅ Comprehensive evaluation metrics
- ✅ Visualization tools

---

## 🏗️ System Architecture

```
basketball_detection_system/
├── train.py                    # Main data preparation pipeline
├── rtdetr_training.py          # RT-DETR training module
├── sam2_tracker.py             # SAM2 tracking module
├── complete_pipeline.py        # Integrated pipeline
├── test_evaluation.py          # Model evaluation
├── inference_deploy.py         # Inference & deployment
└── src/
    ├── download.py             # Data download script
    ├── verify_coco.py          # COCO format verification
    └── split_data.py           # Dataset splitting
```

---

## 📦 Installation

### Requirements

```bash
# Python 3.8+
python >= 3.8

# Core dependencies
torch >= 2.0.0
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
matplotlib >= 3.7.0
pandas >= 2.0.0
pycocotools >= 2.0.6
```

### Setup

```bash
# 1. Clone or download the repository
git clone <your-repo-url>
cd basketball_detection_system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install ultralytics opencv-python numpy matplotlib pandas pycocotools tqdm roboflow

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline from data download to model training:

```python
from complete_pipeline import BasketballTrainingPipeline

# Initialize pipeline
pipeline = BasketballTrainingPipeline(base_dir="./basketball_pipeline")

# Run full pipeline
results = pipeline.run_full_pipeline(
    # Data preparation
    download_data=True,
    train_ratio=0.8,
    val_ratio=0.2,
    
    # Training
    model_size='rtdetr-l',
    epochs=100,
    batch_size=16,
    img_size=640,
    pretrained=True,
    
    # Tracking (optional)
    test_video="./test_video.mp4",
    conf_threshold=0.25
)
```

### Option 2: Step-by-Step

#### Step 1: Data Preparation

```python
from train import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_pipeline(
    download=True,
    train_ratio=0.8,
    val_ratio=0.2
)
```

#### Step 2: Train RT-DETR Model

```python
from rtdetr_training import RTDETRTrainer

trainer = RTDETRTrainer(model_size='rtdetr-l')

# Prepare dataset
data_yaml = trainer.prepare_dataset_yaml("./data/split_dataset")

# Train
training_results = trainer.train(
    data_yaml=data_yaml,
    epochs=100,
    batch_size=16,
    img_size=640,
    pretrained=True
)

# Validate
val_results = trainer.validate(data_yaml=data_yaml)
```

#### Step 3: Track Video

```python
from sam2_tracker import SAM2Tracker

tracker = SAM2Tracker()

tracking_results = tracker.track_video(
    video_path="./test_video.mp4",
    detector_model_path="./runs/train/basketball_rtdetr/weights/best.pt",
    conf_threshold=0.25,
    save_video=True
)

# Analyze tracks
stats = tracker.analyze_tracks(tracking_results)
```

#### Step 4: Run Inference

```python
from inference_deploy import BasketballDetector

detector = BasketballDetector(
    model_path="./runs/train/basketball_rtdetr/weights/best.pt",
    conf_threshold=0.25
)

# Single image
result = detector.detect_image("./test.jpg", save_path="./output.jpg")

# Video
video_result = detector.detect_video(
    video_path="./game.mp4",
    output_path="./game_detected.mp4"
)

# Webcam
detector.detect_webcam(camera_id=0)
```

---

## 📚 Module Documentation

### 1. Data Preparation (`train.py`)

**TrainingPipeline Class**

Main pipeline for data download, verification, splitting, and visualization.

```python
pipeline = TrainingPipeline(base_dir="./data")

# Download and prepare data
results = pipeline.run_pipeline(
    download=True,      # Download from Roboflow
    train_ratio=0.8,    # 80% training
    val_ratio=0.2       # 20% validation
)
```

**Key Methods:**
- `download_data()`: Downloads dataset from Roboflow
- `verify_dataset()`: Verifies COCO format compliance
- `split_dataset()`: Splits data into train/val/test
- `visualize_split()`: Creates visualization plots

### 2. RT-DETR Training (`rtdetr_training.py`)

**RTDETRTrainer Class**

Handles RT-DETR model training and evaluation.

```python
trainer = RTDETRTrainer(
    model_size='rtdetr-l',  # or 'rtdetr-x' for larger model
    device='cuda'            # or 'cpu'
)
```

**Training Parameters:**
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 16)
- `img_size`: Input image size (default: 640)
- `pretrained`: Use pretrained weights (default: True)
- `augment`: Apply data augmentation (default: True)

**Key Methods:**
- `prepare_dataset_yaml()`: Creates dataset config
- `train()`: Trains the model
- `validate()`: Validates on test set
- `predict()`: Run inference
- `export_model()`: Export to ONNX/TorchScript

### 3. SAM2 Tracking (`sam2_tracker.py`)

**SAM2Tracker Class**

Player tracking using detections and IoU matching.

```python
tracker = SAM2Tracker()

tracking_results = tracker.track_video(
    video_path="./video.mp4",
    detector_model_path="./best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45
)
```

**Key Methods:**
- `track_video()`: Track objects in video
- `analyze_tracks()`: Compute track statistics
- `visualize_tracks()`: Create visualization plots

### 4. Evaluation (`test_evaluation.py`)

**ModelEvaluator Class**

Comprehensive model evaluation with COCO metrics.

```python
evaluator = ModelEvaluator(model_path="./best.pt")

results = evaluator.evaluate_test_set(
    test_dir="./data/test",
    conf_threshold=0.25,
    save_predictions=True
)

# Compare multiple models
comparison = evaluator.compare_models(
    model_paths=["./model1.pt", "./model2.pt"],
    test_dir="./data/test"
)
```

**Metrics Computed:**
- mAP (mean Average Precision)
- mAP@50, mAP@75
- Per-class AP
- Precision, Recall
- Detection counts

### 5. Inference & Deployment (`inference_deploy.py`)

**BasketballDetector Class**

Real-time inference on images, videos, and webcams.

```python
detector = BasketballDetector(
    model_path="./best.pt",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Single image
detector.detect_image("./image.jpg", save_path="./output.jpg")

# Video
detector.detect_video("./video.mp4", output_path="./output.mp4")

# Batch processing
detector.batch_process_images("./images/", "./output/")

# Webcam
detector.detect_webcam(camera_id=0)
```

**DeploymentHelper Class**

Tools for model deployment and benchmarking.

```python
# Benchmark model
results = DeploymentHelper.benchmark_model(
    model_path="./best.pt",
    img_size=640,
    num_runs=100
)

# Export model
exports = DeploymentHelper.export_for_deployment(
    model_path="./best.pt",
    formats=['onnx', 'torchscript']
)
```

---

## 🎓 Training Pipeline

### Complete Training Workflow

```python
from complete_pipeline import BasketballTrainingPipeline

pipeline = BasketballTrainingPipeline()

# Full pipeline with all options
results = pipeline.run_full_pipeline(
    # Data preparation
    download_data=True,
    train_ratio=0.8,
    val_ratio=0.2,
    
    # RT-DETR training
    model_size='rtdetr-l',      # Model architecture
    epochs=100,                  # Training epochs
    batch_size=16,               # Batch size
    img_size=640,                # Input size
    pretrained=True,             # Use pretrained weights
    
    # Video tracking
    test_video="./game.mp4",
    conf_threshold=0.25,
    
    # Control
    skip_training=False,         # Skip training if model exists
    model_path=None              # Path to existing model
)
```

### Training Configuration

**Recommended Settings:**

| Configuration | Small Dataset (<500 imgs) | Medium Dataset (500-2000 imgs) | Large Dataset (>2000 imgs) |
|---------------|---------------------------|--------------------------------|----------------------------|
| Model Size    | rtdetr-l                  | rtdetr-l                       | rtdetr-x                   |
| Epochs        | 150                       | 100                            | 80                         |
| Batch Size    | 8                         | 16                             | 32                         |
| Image Size    | 640                       | 640                            | 640-1280                   |
| Patience      | 50                        | 50                             | 30                         |

### Data Augmentation

The pipeline automatically applies:
- HSV color jittering
- Random horizontal flip
- Translation and scaling
- Mosaic augmentation
- MixUp (optional)

---

## 🚀 Inference & Deployment

### 1. Single Image Detection

```python
from inference_deploy import BasketballDetector

detector = BasketballDetector(model_path="./best.pt")

result = detector.detect_image(
    image_path="./test.jpg",
    save_path="./output.jpg",
    show=True
)

print(f"Detections: {result['num_detections']}")
for det in result['detections']:
    print(f"  {det['class']}: {det['confidence']:.2f}")
```

### 2. Video Processing

```python
video_result = detector.detect_video(
    video_path="./game.mp4",
    output_path="./game_detected.mp4",
    display=False,  # Set True to show while processing
    max_frames=None,  # Process all frames
    save_json=True
)

print(f"FPS: {video_result['avg_fps']:.1f}")
print(f"Avg detections/frame: {video_result['avg_detections_per_frame']:.1f}")
```

### 3. Real-time Webcam

```python
# Run real-time detection on webcam
detector.detect_webcam(
    camera_id=0,
    save_output=True,
    output_path="./webcam_recording.mp4"
)
# Press 'q' to quit
```

### 4. Batch Processing

```python
batch_result = detector.batch_process_images(
    image_dir="./test_images",
    output_dir="./output_images",
    save_json=True
)

print(f"Processed {batch_result['num_images']} images")
print(f"Avg time: {batch_result['avg_time_per_image']:.2f}s/image")
```

### 5. Model Export

```python
from inference_deploy import DeploymentHelper

# Export to ONNX for deployment
exports = DeploymentHelper.export_for_deployment(
    model_path="./best.pt",
    output_dir="./deployment",
    formats=['onnx', 'torchscript']
)

# Benchmark performance
benchmark = DeploymentHelper.benchmark_model(
    model_path="./best.pt",
    img_size=640,
    num_runs=100
)

print(f"Mean FPS: {benchmark['mean_fps']:.1f}")
print(f"Mean latency: {benchmark['mean_time']*1000:.2f}ms")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
trainer.train(batch_size=8)  # or 4

# Or reduce image size
trainer.train(img_size=416)  # instead of 640
```

**2. Slow Training**
```python
# Enable mixed precision
trainer.train(amp=True)

# Use fewer workers if CPU bottleneck
trainer.train(workers=4)

# Cache images in memory (if RAM allows)
trainer.train(cache=True)
```

**3. Poor Detection Performance**
```python
# Increase training epochs
trainer.train(epochs=150)

# Try larger model
trainer = RTDETRTrainer(model_size='rtdetr-x')

# Adjust confidence threshold
detector = BasketballDetector(conf_threshold=0.15)  # Lower for more detections
```

**4. Dataset Format Errors**
```python
# Verify COCO format
pipeline = TrainingPipeline()
is_valid = pipeline.verify_dataset("./dataset_path")

# Check for common issues:
# - Missing images
# - Invalid bounding boxes
# - Mismatched annotations
```

---

## ⚡ Performance Optimization

### Training Optimization

1. **Use GPU**: Ensure CUDA is available
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

2. **Optimize Batch Size**: Find maximum batch size that fits in GPU memory
```python
# Start with 32 and reduce if OOM
batch_sizes = [32, 16, 8, 4]
for bs in batch_sizes:
    try:
        trainer.train(batch_size=bs, epochs=1)
        print(f"Batch size {bs} works!")
        break
    except RuntimeError:
        continue
```

3. **Mixed Precision**: Enable AMP for faster training
```python
trainer.train(amp=True)  # 2-3x speedup on modern GPUs
```

### Inference Optimization

1. **Export to ONNX**: Faster inference
```python
onnx_path = trainer.export_model(format='onnx')
# Use ONNX runtime for inference
```

2. **Batch Inference**: Process multiple images together
```python
# More efficient than processing one-by-one
detector.batch_process_images("./images", "./output")
```

3. **Reduce Image Size**: Lower resolution for faster inference
```python
detector = BasketballDetector(
    model_path="./best.pt",
    img_size=416  # Instead of 640
)
```

4. **Optimize Confidence Threshold**: Balance speed vs accuracy
```python
# Higher threshold = fewer detections = faster
detector = BasketballDetector(conf_threshold=0.4)
```

### Hardware Recommendations

| Use Case | Recommended Hardware | Expected FPS |
|----------|---------------------|--------------|
| Training | RTX 3080+ or better | N/A |
| Real-time Inference (640px) | RTX 2060+ | 30-60 |
| Real-time Inference (416px) | GTX 1660+ | 30-60 |
| Batch Processing | Any GPU | Varies |
| CPU Only | Modern CPU | 5-10 |

---

## 📊 Model Performance

### Expected Metrics

On a well-labeled basketball dataset:

| Metric | Expected Range | Good Performance |
|--------|----------------|------------------|
| mAP@50 | 0.60-0.85 | > 0.75 |
| mAP@50:95 | 0.40-0.70 | > 0.55 |
| Precision | 0.70-0.90 | > 0.80 |
| Recall | 0.65-0.85 | > 0.75 |
| Inference Speed | 20-60 FPS | > 30 FPS |

### Improving Model Performance

1. **More Training Data**: Collect and label more images
2. **Data Augmentation**: Already enabled in pipeline
3. **Longer Training**: Increase epochs (100-150)
4. **Larger Model**: Use rtdetr-x instead of rtdetr-l
5. **Fine-tuning**: Adjust hyperparameters based on validation metrics

---

## 📁 Output Directory Structure

After running the full pipeline:

```
basketball_pipeline/
├── data/
│   ├── split_dataset/
│   │   ├── train/
│   │   │   ├── _annotations.coco.json
│   │   │   └── images/
│   │   ├── val/
│   │   │   ├── _annotations.coco.json
│   │   │   └── images/
│   │   └── dataset.yaml
│   └── dataset_split_analysis.png
├── runs/
│   └── train/
│       └── basketball_rtdetr/
│           ├── weights/
│           │   ├── best.pt       # Best model checkpoint
│           │   └── last.pt       # Last epoch checkpoint
│           ├── results.png       # Training curves
│           ├── confusion_matrix.png
│           └── args.yaml         # Training configuration
├── tracking_output/
│   ├── video_tracked.mp4
│   ├── video_tracking.json
│   └── track_analysis.png
├── evaluation_output/
│   ├── predictions.json
│   ├── evaluation_results.json
│   └── evaluation_plots.png
└── pipeline_results.json
```

---

## 🔧 Advanced Usage

### Custom Training Configuration

```python
from rtdetr_training import RTDETRTrainer

trainer = RTDETRTrainer(model_size='rtdetr-l')

# Advanced training with custom hyperparameters
results = trainer.train(
    data_yaml="./dataset.yaml",
    epochs=150,
    batch_size=16,
    img_size=640,
    
    # Optimizer settings
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    
    # Training control
    patience=50,
    cos_lr=True,
    close_mosaic=10,
    
    # Other
    workers=8,
    seed=42,
    amp=True
)
```

### Custom Tracking Parameters

```python
from sam2_tracker import SAM2Tracker

tracker = SAM2Tracker()

# Fine-tune tracking parameters
tracking_results = tracker.track_video(
    video_path="./game.mp4",
    detector_model_path="./best.pt",
    conf_threshold=0.25,      # Lower = more detections
    iou_threshold=0.45,       # Higher = stricter matching
    max_frames=1000,          # Limit frames for testing
    save_video=True,
    save_json=True
)

# Custom track analysis
stats = tracker.analyze_tracks(tracking_results)
for track_id, track_stats in stats['tracks_stats'].items():
    if track_stats['duration_seconds'] > 5.0:  # Only long tracks
        print(f"Track {track_id}: {track_stats['duration_seconds']:.1f}s")
```

### Multi-Model Comparison

```python
from test_evaluation import ModelEvaluator

evaluator = ModelEvaluator(model_path="./best.pt")

# Compare different models
model_paths = [
    "./runs/train/exp1/weights/best.pt",  # rtdetr-l, 100 epochs
    "./runs/train/exp2/weights/best.pt",  # rtdetr-l, 150 epochs
    "./runs/train/exp3/weights/best.pt",  # rtdetr-x, 100 epochs
]

comparison_df = evaluator.compare_models(
    model_paths=model_paths,
    test_dir="./data/split_dataset/test",
    output_dir="./comparison"
)

print(comparison_df)
```

---

## 🎯 Use Cases

### 1. Player Statistics

Track individual players and compute statistics:

```python
tracker = SAM2Tracker()
results = tracker.track_video("./game.mp4", "./best.pt")
stats = tracker.analyze_tracks(results)

for track_id, track_stats in stats['tracks_stats'].items():
    print(f"Player {track_id}:")
    print(f"  Time on court: {track_stats['duration_seconds']:.1f}s")
    print(f"  Movement: {track_stats['total_movement']:.1f} pixels")
    print(f"  Avg speed: {track_stats['avg_speed']:.2f} px/frame")
```

### 2. Jersey Number Recognition

If your model includes jersey number detection:

```python
detector = BasketballDetector(model_path="./best.pt")
result = detector.detect_image("./player.jpg")

for det in result['detections']:
    if det['class'] == 'jersey_number':
        print(f"Jersey #{det['number']}: {det['confidence']:.2f}")
```

### 3. Real-time Game Analysis

Process live game feed:

```python
detector = BasketballDetector(model_path="./best.pt")

# Process RTSP stream
detector.detect_video(
    video_path="rtsp://camera_ip/stream",
    output_path="./live_analysis.mp4",
    display=True
)
```

### 4. Training Data Collection

Use the model to help label new data:

```python
detector = BasketballDetector(
    model_path="./best.pt",
    conf_threshold=0.5  # High confidence only
)

# Pre-label new images
batch_result = detector.batch_process_images(
    image_dir="./unlabeled_images",
    output_dir="./pre_labeled",
    save_json=True
)

# Use JSON results as starting point for manual labeling
```

---

## 📚 Additional Resources

### Documentation
- [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Tutorials
- [Object Detection Basics](https://docs.ultralytics.com/tasks/detect/)
- [Model Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)
- [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

### Community
- [Ultralytics Discord](https://discord.com/invite/ultralytics)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@software{basketball_detection_2024,
  title={Basketball Player Detection and Tracking System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone <your-repo>
cd basketball_detection_system

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the [Troubleshooting](#troubleshooting) section
- Review closed issues for similar problems

---

## 🗺️ Roadmap

- [ ] SAM2 integration (when publicly released)
- [ ] Multi-camera support
- [ ] Player re-identification across cameras
- [ ] Advanced jersey number OCR
- [ ] Team classification
- [ ] Play analysis and statistics
- [ ] Web interface for visualization
- [ ] Mobile deployment (iOS/Android)

---

## ⚠️ Known Limitations

1. **SAM2**: Currently using placeholder - will integrate when SAM2 is publicly released
2. **Tracking**: Simple IoU-based tracking - can be improved with more sophisticated algorithms
3. **Jersey Numbers**: OCR for jersey numbers requires additional fine-tuning
4. **Occlusions**: Performance degrades with heavy player occlusion
5. **Lighting**: Requires good lighting conditions for optimal performance

---

## 📊 Version History

### v1.0.0 (Current)
- Initial release
- RT-DETR training pipeline
- Basic tracking functionality
- Inference and deployment tools
- Comprehensive evaluation metrics

### Planned Updates
- v1.1.0: SAM2 integration
- v1.2.0: Enhanced tracking algorithms
- v1.3.0: Multi-camera support
- v2.0.0: Web interface and API

---

**Last Updated**: 2024
**Maintainer**: Your Name
**Status**: Active Development