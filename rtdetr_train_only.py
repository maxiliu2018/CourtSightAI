"""
RT-DETR Training Module for Player and Number Detection
Training functions only - separated from validation/testing
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from ultralytics import RTDETR
import json
from datetime import datetime


class RTDETRTrainer:
    """RT-DETR model trainer for basketball player and number detection"""
    
    def __init__(self, 
                 model_size: str = 'rtdetr-l',
                 device: Optional[str] = None):
        """
        Initialize RT-DETR trainer
        
        Args:
            model_size: Model size ('rtdetr-l', 'rtdetr-x')
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_size = model_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.training_results = {}
        
        print(f"🎯 Initializing RT-DETR Trainer")
        print(f"   Model: {model_size}")
        print(f"   Device: {self.device}")
        
    def _coco_to_yolo_bbox(self, coco_bbox, img_width, img_height):
        """Convert COCO bbox to YOLO format"""
        x, y, w, h = coco_bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        # Clip to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        return [x_center, y_center, w_norm, h_norm]
    
    def _convert_coco_to_yolo(self, coco_dir: str, splits=['train', 'val', 'test']):
        """
        Convert COCO format to YOLO format in-place
        
        Args:
            coco_dir: Directory containing COCO format data
            splits: List of splits to convert
        """
        coco_path = Path(coco_dir)
        
        print(f"\n🔄 Converting COCO to YOLO format...")
        
        category_mapping = {}
        category_names = {}
        
        for split in splits:
            split_dir = coco_path / split
            ann_file = split_dir / '_annotations.coco.json'
            
            if not ann_file.exists():
                continue
            
            print(f"   Converting {split}...")
            
            # Load COCO annotations
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create labels directory
            labels_dir = split_dir / 'labels'
            labels_dir.mkdir(exist_ok=True)
            
            # Build category mapping
            if not category_mapping:
                categories = sorted(coco_data['categories'], key=lambda x: x['id'])
                category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
                category_names = {idx: cat['name'] for idx, cat in enumerate(categories)}
            
            # Build image info
            image_info = {img['id']: img for img in coco_data['images']}
            
            # Group annotations by image
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
            
            # Convert each image's annotations
            for img_id, img_data in image_info.items():
                img_filename = img_data['file_name']
                img_width = img_data['width']
                img_height = img_data['height']
                
                # Create label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = labels_dir / label_filename
                
                # Get annotations
                anns = annotations_by_image.get(img_id, [])
                
                if not anns:
                    label_path.touch()
                    continue
                
                # Convert to YOLO format
                yolo_lines = []
                for ann in anns:
                    coco_cat_id = ann['category_id']
                    yolo_class_idx = category_mapping[coco_cat_id]
                    coco_bbox = ann['bbox']
                    yolo_bbox = self._coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                    yolo_line = f"{yolo_class_idx} {' '.join(f'{x:.6f}' for x in yolo_bbox)}"
                    yolo_lines.append(yolo_line)
                
                # Write label file
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
            
            print(f"   ✅ {split}: {len(image_info)} images converted")
        
        return category_names
    
    def prepare_dataset_yaml(self, split_dir: str, output_path: str = None) -> str:
        """
        Create dataset YAML file for YOLO/RT-DETR training
        Automatically converts COCO to YOLO format if needed
        
        Args:
            split_dir: Directory containing split dataset
            output_path: Path to save YAML file
            
        Returns:
            Path to created YAML file
        """
        split_path = Path(split_dir)
        
        print(f"\n📋 Preparing dataset for training...")
        
        # Load train annotations to get class names
        train_json = split_path / 'train' / '_annotations.coco.json'
        with open(train_json, 'r') as f:
            coco_data = json.load(f)
        
        # Extract categories
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        class_names = [categories[i] for i in sorted(categories.keys())]
        
        # Check if YOLO labels exist, if not convert
        train_labels_dir = split_path / 'train' / 'labels'
        if not train_labels_dir.exists() or not list(train_labels_dir.glob('*.txt')):
            print(f"\n⚠️  YOLO labels not found. Converting from COCO format...")
            category_names = self._convert_coco_to_yolo(str(split_path))
            print(f"✅ Conversion complete!")
        else:
            print(f"✅ YOLO labels already exist")
        
        # Check directory structure and reorganize if needed
        train_dir = split_path / 'train'
        images_dir = train_dir / 'images'
        
        # If images/ subdirectory doesn't exist, images are in the root
        if not images_dir.exists():
            print(f"\n🔧 Reorganizing directory structure...")
            self._reorganize_directory_structure(str(split_path))
            print(f"✅ Directory structure updated")
        
        # Create YAML configuration
        yaml_config = {
            'path': str(split_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images' if (split_path / 'test' / 'images').exists() else '',
            'names': {i: name for i, name in enumerate(class_names)},
            'nc': len(class_names)
        }
        
        # Save YAML
        if output_path is None:
            output_path = split_path / 'dataset.yaml'
        else:
            output_path = Path(output_path)
            
        with open(output_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Dataset YAML created: {output_path}")
        print(f"   Classes ({len(class_names)}): {', '.join(class_names)}")
        
        return str(output_path)
    
    def _reorganize_directory_structure(self, split_dir: str):
        """
        Reorganize directory to have images/ and labels/ subdirectories
        Expected structure:
        split_dir/
          train/
            images/
            labels/
          val/
            images/
            labels/
        """
        import shutil
        
        split_path = Path(split_dir)
        
        for split in ['train', 'val', 'test']:
            split_dir = split_path / split
            if not split_dir.exists():
                continue
            
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            # If images/ doesn't exist, create it and move images
            if not images_dir.exists():
                print(f"   Reorganizing {split}/...")
                images_dir.mkdir(exist_ok=True)
                
                # Move all image files to images/
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
                for ext in image_extensions:
                    for img_file in split_dir.glob(f'*{ext}'):
                        if img_file.is_file():
                            shutil.move(str(img_file), str(images_dir / img_file.name))
                
                print(f"   ✅ {split}: Moved images to images/ subdirectory")
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              project: str = './runs/train',
              name: str = 'rtdetr_basketball',
              pretrained: bool = True,
              augment: bool = True,
              lr0=0.001, 
              optimizer='SGD',
              **kwargs) -> Dict:
        """
        Train RT-DETR model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            project: Project directory
            name: Experiment name
            pretrained: Use pretrained weights
            augment: Apply data augmentation
            lr0: Initial learning rate
            optimizer: Optimizer type ('SGD', 'AdamW', etc.)
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary with training results
        """
        print("\n" + "="*60)
        print("🚀 Starting RT-DETR Training")
        print("="*60)
        
        # Initialize model
        if pretrained:
            self.model = RTDETR(f'{self.model_size}.pt')
            print(f"✅ Loaded pretrained {self.model_size} model")
        else:
            self.model = RTDETR(f'{self.model_size}.yaml')
            print(f"✅ Initialized {self.model_size} model from scratch")
        
        # Training configuration
        train_config = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'project': project,
            'name': name,
            'device': self.device,
            'workers': 8,
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': False,
            'exist_ok': True,
            'pretrained': pretrained,
            'optimizer': optimizer,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'lr0': lr0,
        }
        
        # Data augmentation settings
        if augment:
            train_config.update({
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
            })
        
        # Update with any additional kwargs
        train_config.update(kwargs)
        
        print("\n📋 Training Configuration:")
        for key, value in train_config.items():
            print(f"   {key}: {value}")
        
        # Start training
        print("\n🎯 Training started...")
        results = self.model.train(**train_config)
        
        # Store results
        self.training_results = {
            'model_path': str(Path(project) / name / 'weights' / 'best.pt'),
            'last_model_path': str(Path(project) / name / 'weights' / 'last.pt'),
            'results_dir': str(Path(project) / name),
            'final_metrics': self._extract_metrics(results),
            'config': train_config
        }
        
        # Save training info for later use
        self._save_training_info(self.training_results)
        
        print("\n" + "="*60)
        print("✨ Training Complete!")
        print("="*60)
        print(f"📁 Results saved to: {self.training_results['results_dir']}")
        print(f"🏆 Best model: {self.training_results['model_path']}")
        print(f"📝 Training info saved for validation script")
        
        return self.training_results
    
    def _extract_metrics(self, results) -> Dict:
        """Extract key metrics from training results"""
        try:
            metrics = {}
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
            elif isinstance(results, dict):
                metrics = results
            return metrics
        except:
            return {}
    
    def _save_training_info(self, training_results: Dict):
        """Save training information for validation script"""
        info_path = Path(training_results['results_dir']) / 'training_info.yaml'
        
        # Extract config without non-serializable objects
        config_to_save = {}
        for k, v in training_results['config'].items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config_to_save[k] = v
        
        info = {
            'model_path': training_results['model_path'],
            'last_model_path': training_results['last_model_path'],
            'results_dir': training_results['results_dir'],
            'model_size': self.model_size,
            'device': self.device,
            'data_yaml': config_to_save.get('data'),
            'epochs': config_to_save.get('epochs'),
            'batch_size': config_to_save.get('batch'),
            'img_size': config_to_save.get('imgsz'),
            'trained_at': datetime.now().isoformat(),
        }
        
        with open(info_path, 'w') as f:
            yaml.dump(info, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Training info saved to: {info_path}")


def main():
    """Example training usage"""
    
    # Configuration
    SPLIT_DIR = "./data/split_dataset"
    PROJECT_DIR = "./runs/train"
    EXPERIMENT_NAME = "basketball_rtdetr"
    
    # Initialize trainer
    trainer = RTDETRTrainer(model_size='rtdetr-l')
    
    # Prepare dataset YAML
    data_yaml = trainer.prepare_dataset_yaml(SPLIT_DIR)
    
    # Train model
    results = trainer.train(
        data_yaml=data_yaml,
        epochs=100,
        batch_size=16,
        img_size=640,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        pretrained=True,
        augment=True,
        lr0=0.001,
        optimizer='AdamW'
    )
    
    print("\n" + "="*60)
    print("✨ Training Complete!")
    print("="*60)
    print(f"\n📁 Model saved at: {results['model_path']}")
    print(f"📊 Results directory: {results['results_dir']}")
    print(f"\n💡 Next step: Run rtdetr_validation.py to validate your model!")
    
    return results


if __name__ == "__main__":
    main()
