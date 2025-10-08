"""
Basketball RT-DETR Validation Pipeline
Handles validation on validation and test sets
"""

import os
import torch
import yaml
from pathlib import Path
from typing import Dict, Optional
from ultralytics import RTDETR


class BasketballValidationPipeline:
    """Validation pipeline for basketball player and number detection"""
    
    def __init__(self, 
                 model_path: str,
                 data_yaml: str,
                 device: Optional[str] = None):
        """
        Initialize validation pipeline
        
        Args:
            model_path: Path to trained model weights
            data_yaml: Path to dataset YAML configuration
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"🎯 Initializing Basketball Validation Pipeline")
        print(f"   Model: {model_path}")
        print(f"   Dataset: {data_yaml}")
        print(f"   Device: {self.device}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"❌ Model not found at: {self.model_path}")
        
        print(f"\n📥 Loading model from: {self.model_path}")
        self.model = RTDETR(self.model_path)
        print(f"✅ Loaded model from: {self.model_path}")
    
    def apply_validation_fix(self):
        """Apply fix for RT-DETR validation ratio_pad bug"""
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.utils.ops import scale_boxes
        
        # Store the original method
        original_scale_preds = DetectionValidator.scale_preds
        
        # Create the patched method
        def patched_scale_preds(validator_self, preds, pbatch, orig_imgs):
            """Scale predictions with original image dimensions."""
            predn = preds.clone()
            
            # Get original dimensions from orig_imgs or batch
            if orig_imgs is not None and len(orig_imgs) > 0:
                if hasattr(orig_imgs[0], 'shape'):
                    ori_shape = orig_imgs[0].shape[:2]
                else:
                    ori_shape = preds.shape[-2:]
            else:
                ori_shape = pbatch.get('ori_shape', pbatch.get('resized_shape', preds.shape[-2:]))
            
            # Scale boxes from inference size to original image size
            predn[:, :4] = scale_boxes(
                preds.shape[-2:],  # from shape (inference size)
                predn[:, :4],  # boxes to scale
                ori_shape,  # to shape (original size)
                ratio_pad=pbatch.get('ratio_pad')
            )
            return predn
        
        # Apply the monkey patch
        DetectionValidator.scale_preds = patched_scale_preds
        print("✅ Applied RT-DETR validation fix")
    
    def run_validation_pipeline(self,
                               img_size: int = 640,
                               batch_size: int = 16,
                               conf: float = 0.001,
                               iou: float = 0.6) -> Dict:
        """
        Run complete validation pipeline
        
        Args:
            img_size: Image size for validation
            batch_size: Batch size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            Dictionary with validation and test results
        """
        print("\n" + "="*70)
        print("🔍 STARTING VALIDATION PIPELINE")
        print("="*70)
        
        # Validation configuration
        val_config = {
            'data': self.data_yaml,
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
        
        # Run validation on validation set
        print("\n📊 Running validation on VALIDATION SET...")
        print("-" * 70)
        val_results = self.model.val(**val_config)
        
        # Print validation results
        print("\n✅ VALIDATION SET RESULTS:")
        if hasattr(val_results, 'box'):
            print(f"   mAP50-95: {val_results.box.map:.4f}")
            print(f"   mAP50: {val_results.box.map50:.4f}")
            print(f"   mAP75: {val_results.box.map75:.4f}")
            print(f"   Precision: {val_results.box.mp:.4f}")
            print(f"   Recall: {val_results.box.mr:.4f}")
        
        # Run validation on test set
        print("\n📊 Running validation on TEST SET...")
        print("-" * 70)
        
        # Load data.yaml to get test path
        with open(self.data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create temporary yaml with test path
        test_yaml = str(Path(self.data_yaml).parent / 'test_data.yaml')
        test_data = data.copy()
        test_data['val'] = data.get('test', data['val'])  # Use test set as validation
        
        with open(test_yaml, 'w') as f:
            yaml.dump(test_data, f, default_flow_style=False)
        
        # Update config for test set
        test_config = val_config.copy()
        test_config['data'] = test_yaml
        
        test_results = self.model.val(**test_config)
        
        # Print test results
        print("\n✅ TEST SET RESULTS:")
        if hasattr(test_results, 'box'):
            print(f"   mAP50-95: {test_results.box.map:.4f}")
            print(f"   mAP50: {test_results.box.map50:.4f}")
            print(f"   mAP75: {test_results.box.map75:.4f}")
            print(f"   Precision: {test_results.box.mp:.4f}")
            print(f"   Recall: {test_results.box.mr:.4f}")
        
        # Clean up temporary yaml
        if os.path.exists(test_yaml):
            os.remove(test_yaml)
        
        # Print summary
        print("\n" + "="*70)
        print("✅ VALIDATION PIPELINE COMPLETE!")
        print("="*70)
        print("\n📊 SUMMARY:")
        print(f"\nValidation Set:")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"\nTest Set:")
        print(f"   mAP50-95: {test_results.box.map:.4f}")
        print(f"   mAP50: {test_results.box.map50:.4f}")
        
        return {
            'validation': val_results,
            'test': test_results
        }


def main():
    """Main validation execution"""
    
    # Configuration - UPDATE THESE PATHS
    MODEL_PATH = "basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt"
    DATA_YAML = "/content/CourtSightAI/data/split_dataset/dataset.yaml"
    
    # Initialize pipeline
    pipeline = BasketballValidationPipeline(
        model_path=MODEL_PATH,
        data_yaml=DATA_YAML
    )
    
    # Apply validation fix
    pipeline.apply_validation_fix()
    
    # Run validation pipeline
    results = pipeline.run_validation_pipeline(
        img_size=640,
        batch_size=16,
        conf=0.001,
        iou=0.6
    )
    
    return results


if __name__ == "__main__":
    results = main()