"""
RT-DETR Validation and Testing Module
Validation, testing, and inference functions - separated from training
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from ultralytics import RTDETR
import cv2
import matplotlib.pyplot as plt


class RTDETRValidator:
    """RT-DETR model validator and tester for basketball player and number detection"""
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None):
        """
        Initialize RT-DETR validator
        
        Args:
            model_path: Path to trained model weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_info = None
        
        print(f"🎯 Initializing RT-DETR Validator")
        print(f"   Model: {model_path}")
        print(f"   Device: {self.device}")
        
        # Load model
        self._load_model()
        
        # Load training info if available
        self._load_training_info()
    
    def _load_model(self):
        """Load the trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"❌ Model not found at: {self.model_path}")
        
        print(f"📥 Loading model...")
        self.model = RTDETR(self.model_path)
        print(f"✅ Model loaded successfully!")
    
    def _load_training_info(self):
        """Load training information if available"""
        # Try to find training_info.yaml in parent directories
        model_dir = Path(self.model_path).parent.parent
        info_path = model_dir / 'training_info.yaml'
        
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.training_info = yaml.safe_load(f)
            print(f"✅ Loaded training info from: {info_path}")
            print(f"   Trained at: {self.training_info.get('trained_at', 'N/A')}")
            print(f"   Epochs: {self.training_info.get('epochs', 'N/A')}")
        else:
            print(f"⚠️  Training info not found at: {info_path}")
    
    def apply_validation_fix(self):
        """Apply fix for RT-DETR validation ratio_pad bug"""
        import ultralytics.models.yolo.detect.val as detect_val
        
        original_scale_preds = detect_val.DetectionValidator.scale_preds
        
        def patched_scale_preds(self, preds, img, orig_imgs, ratio_pad=None):
            # Convert ratio_pad to expected format if it's a float
            if ratio_pad is not None and isinstance(ratio_pad, (int, float)):
                ratio_pad = [[ratio_pad], [0, 0]]
            return original_scale_preds(self, preds, img, orig_imgs, ratio_pad)
        
        detect_val.DetectionValidator.scale_preds = patched_scale_preds
        print("✅ Applied RT-DETR validation fix")
    
    def validate(self, 
                 data_yaml: str,
                 img_size: int = 640,
                 batch_size: int = 16,
                 conf: float = 0.001,
                 iou: float = 0.6,
                 **kwargs) -> Dict:
        """
        Validate trained model on validation set
        
        Args:
            data_yaml: Path to dataset YAML
            img_size: Image size for validation
            batch_size: Batch size
            conf: Confidence threshold
            iou: IoU threshold
            **kwargs: Additional validation arguments
            
        Returns:
            Validation results
        """
        print("\n" + "="*60)
        print("🔍 Running Validation on Validation Set")
        print("="*60)
        
        # Validate
        val_config = {
            'data': data_yaml,
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'workers': 8,
            'verbose': True,
            'save_json': False,
            'save_hybrid': False,
            'conf': conf,
            'iou': iou,
            'max_det': 300,
            'plots': True,
            'rect': False,  # Important for RT-DETR
        }
        val_config.update(kwargs)
        
        results = self.model.val(**val_config)
        
        print("\n📊 Validation Results:")
        if hasattr(results, 'box'):
            print(f"   mAP50: {results.box.map50:.4f}")
            print(f"   mAP50-95: {results.box.map:.4f}")
            print(f"   Precision: {results.box.mp:.4f}")
            print(f"   Recall: {results.box.mr:.4f}")
        
        return results
    
    def test(self,
             data_yaml: str,
             img_size: int = 640,
             batch_size: int = 16,
             conf: float = 0.001,
             iou: float = 0.6,
             **kwargs) -> Dict:
        """
        Test model on test set
        
        Args:
            data_yaml: Path to dataset YAML
            img_size: Image size
            batch_size: Batch size
            conf: Confidence threshold
            iou: IoU threshold
            **kwargs: Additional test arguments
            
        Returns:
            Test results
        """
        print("\n" + "="*60)
        print("🧪 Running Evaluation on Test Set")
        print("="*60)
        
        # Load data.yaml to get test path
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create temporary yaml with test path
        test_yaml = str(Path(data_yaml).parent / 'test_data.yaml')
        test_data = data.copy()
        test_data['val'] = data.get('test', data['val'])  # Use test set as validation
        
        with open(test_yaml, 'w') as f:
            yaml.dump(test_data, f, default_flow_style=False)
        
        # Run validation on test set
        test_config = {
            'data': test_yaml,
            'imgsz': img_size,
            'batch': batch_size,
            'device': self.device,
            'workers': 8,
            'verbose': True,
            'save_json': False,
            'save_hybrid': False,
            'conf': conf,
            'iou': iou,
            'max_det': 300,
            'plots': True,
            'rect': False,
        }
        test_config.update(kwargs)
        
        results = self.model.val(**test_config)
        
        print("\n📊 Test Results:")
        if hasattr(results, 'box'):
            print(f"   mAP50: {results.box.map50:.4f}")
            print(f"   mAP50-95: {results.box.map:.4f}")
            print(f"   Precision: {results.box.mp:.4f}")
            print(f"   Recall: {results.box.mr:.4f}")
        
        # Clean up temporary yaml
        if os.path.exists(test_yaml):
            os.remove(test_yaml)
        
        return results
    
    def predict(self,
                source: str,
                conf: float = 0.25,
                iou: float = 0.45,
                img_size: int = 640,
                save: bool = True,
                save_txt: bool = False,
                save_conf: bool = False,
                project: str = './runs/predict',
                name: str = 'exp',
                **kwargs) -> list:
        """
        Run inference on images/videos
        
        Args:
            source: Path to image/video/directory
            conf: Confidence threshold
            iou: IoU threshold for NMS
            img_size: Image size
            save: Save results
            save_txt: Save results as txt
            save_conf: Save confidences in txt
            project: Project directory
            name: Experiment name
            **kwargs: Additional prediction arguments
            
        Returns:
            List of prediction results
        """
        print("\n" + "="*60)
        print(f"🖼️  Running Predictions on: {source}")
        print("="*60)
        
        # Prediction configuration
        pred_config = {
            'source': source,
            'conf': conf,
            'iou': iou,
            'imgsz': img_size,
            'device': self.device,
            'save': save,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'project': project,
            'name': name,
            'exist_ok': True,
            'line_width': 2,
            'show_labels': True,
            'show_conf': True,
            'max_det': 300,
            'augment': False,
            'agnostic_nms': False,
            'retina_masks': False,
        }
        pred_config.update(kwargs)
        
        # Run prediction
        results = self.model.predict(**pred_config)
        
        print(f"\n✅ Predictions complete")
        print(f"   Processed: {len(results)} images/frames")
        if save:
            print(f"   Results saved to: {project}/{name}")
        
        return results
    
    def predict_samples(self, 
                       data_yaml: str,
                       num_samples: int = 5,
                       split: str = 'val',
                       conf: float = 0.25,
                       **kwargs) -> List:
        """
        Run inference on sample images from dataset
        
        Args:
            data_yaml: Path to dataset YAML
            num_samples: Number of sample images
            split: Which split to use ('train', 'val', 'test')
            conf: Confidence threshold
            **kwargs: Additional prediction arguments
            
        Returns:
            List of prediction results
        """
        print("\n" + "="*60)
        print(f"🖼️  Testing on {num_samples} Sample Images from {split} set")
        print("="*60)
        
        # Load data.yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get images path
        images_path = Path(data['path']) / data[split]
        if not images_path.exists():
            images_path = Path(data[split])
        
        # Get sample images
        sample_images = list(images_path.glob('*.jpg'))[:num_samples]
        
        if not sample_images:
            print("❌ No sample images found!")
            return []
        
        results_list = []
        for img_path in sample_images:
            print(f"\n📸 Processing: {img_path.name}")
            
            # Run inference
            results = self.model.predict(
                source=str(img_path),
                conf=conf,
                device=self.device,
                save=True,
                save_txt=False,
                save_conf=True,
                project='./runs/detect',
                name='test_samples',
                exist_ok=True,
                **kwargs
            )
            
            results_list.append(results[0])
            
            # Print detections
            if len(results[0].boxes) > 0:
                print(f"   ✅ Detected {len(results[0].boxes)} object(s)")
                for box in results[0].boxes:
                    conf_val = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = self.model.names[cls]
                    print(f"      - {cls_name} (Confidence: {conf_val:.3f})")
            else:
                print(f"   ⚠️  No detections")
        
        print(f"\n✅ Sample predictions saved to: ./runs/detect/test_samples")
        
        return results_list
    
    def visualize_predictions(self, results_list: List, num_display: int = 3):
        """
        Display prediction results
        
        Args:
            results_list: List of prediction results
            num_display: Number of results to display
        """
        print("\n📊 Displaying prediction samples...")
        
        num_display = min(num_display, len(results_list))
        fig, axes = plt.subplots(1, num_display, figsize=(15, 5))
        
        if num_display == 1:
            axes = [axes]
        
        for idx, result in enumerate(results_list[:num_display]):
            # Get the plotted image
            img_with_boxes = result.plot()
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f'Sample {idx+1}')
        
        plt.tight_layout()
        plt.show()
    
    def export_model(self,
                    format: str = 'onnx',
                    img_size: int = 640,
                    **kwargs) -> str:
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'engine', etc.)
            img_size: Image size
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        export_config = {
            'format': format,
            'imgsz': img_size,
            'device': self.device,
            'half': False,
            'int8': False,
            'dynamic': False,
            'simplify': True,
            'opset': 17,
        }
        export_config.update(kwargs)
        
        print(f"\n📦 Exporting model to {format.upper()} format...")
        exported_path = self.model.export(**export_config)
        
        print(f"✅ Model exported to: {exported_path}")
        
        return str(exported_path)
    
    def print_model_info(self):
        """Print model information"""
        print("\n" + "="*60)
        print("📋 Model Information")
        print("="*60)
        
        print(f"\nModel Path: {self.model_path}")
        print(f"Number of Classes: {len(self.model.names)}")
        print(f"Class Names: {self.model.names}")
        
        if self.training_info:
            print(f"\nTraining Configuration:")
            print(f"   Model Size: {self.training_info.get('model_size', 'N/A')}")
            print(f"   Epochs: {self.training_info.get('epochs', 'N/A')}")
            print(f"   Batch Size: {self.training_info.get('batch_size', 'N/A')}")
            print(f"   Image Size: {self.training_info.get('img_size', 'N/A')}")
            print(f"   Trained At: {self.training_info.get('trained_at', 'N/A')}")


def main():
    """Example validation usage"""
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "./runs/train/basketball_rtdetr/weights/best.pt"
    DATA_YAML = "./data/split_dataset/dataset.yaml"
    
    # Initialize validator
    validator = RTDETRValidator(model_path=MODEL_PATH)
    
    # Apply validation fix for RT-DETR
    validator.apply_validation_fix()
    
    # Print model info
    validator.print_model_info()
    
    # Run validation
    print("\n" + "="*60)
    print("Starting Validation and Testing Pipeline")
    print("="*60)
    
    val_results = validator.validate(
        data_yaml=DATA_YAML,
        batch_size=16,
        img_size=640,
        conf=0.001,
        iou=0.6
    )
    
    # Run test evaluation
    test_results = validator.test(
        data_yaml=DATA_YAML,
        batch_size=16,
        img_size=640,
        conf=0.001,
        iou=0.6
    )
    
    # Test on sample images
    sample_results = validator.predict_samples(
        data_yaml=DATA_YAML,
        num_samples=5,
        split='val',
        conf=0.25
    )
    
    # Visualize samples
    if sample_results:
        validator.visualize_predictions(sample_results, num_display=3)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ ALL VALIDATION & TESTING COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Validation mAP50: {val_results.box.map50:.4f}")
    print(f"   Validation mAP50-95: {val_results.box.map:.4f}")
    print(f"   Test mAP50: {test_results.box.map50:.4f}")
    print(f"   Test mAP50-95: {test_results.box.map:.4f}")
    
    return {
        'validation': val_results,
        'test': test_results,
        'samples': sample_results
    }


def quick_test_single_image(model_path: str, image_path: str, conf: float = 0.25):
    """
    Quick test on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        conf: Confidence threshold
    """
    validator = RTDETRValidator(model_path=model_path)
    
    results = validator.predict(
        source=image_path,
        conf=conf,
        save=True,
        project='./runs/detect',
        name='quick_test',
        exist_ok=True
    )
    
    # Display result
    if results:
        img_with_boxes = results[0].plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title('Detection Result')
        plt.show()
    
    return results[0] if results else None


if __name__ == "__main__":
    # Run full validation and testing
    results = main()
    
    # Optional: Uncomment to test a specific image
    # quick_test_single_image(
    #     model_path="./runs/train/basketball_rtdetr/weights/best.pt",
    #     image_path="path/to/your/image.jpg",
    #     conf=0.25
    # )