"""
Basketball Player Detection Validation and Tracking Pipeline
Part 2: Model validation, testing, and video tracking
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

# Import your existing modules
try:
    from rtdetr_training import RTDETRTrainer
    from sam2_tracker import SAM2Tracker
except ImportError:
    print("⚠️  Warning: Could not import all modules. Make sure files are in the same directory.")


class BasketballValidationPipeline:
    """
    Validation and tracking pipeline for basketball player detection
    Handles model validation, testing, and video tracking
    """
    
    def __init__(self, base_dir: str = "./basketball_pipeline"):
        self.base_dir = Path(base_dir)
        
        if not self.base_dir.exists():
            raise ValueError(f"Base directory not found: {self.base_dir}. Run training first!")
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'validation': {},
            'tracking': {},
        }
        
        # Load training results if available
        self.training_results = self._load_training_results()
        
        print("🏀 Basketball Player Detection Validation & Tracking Pipeline")
        print("="*60)
        print(f"Working directory: {self.base_dir}")
        print("="*60)
    
    def _load_training_results(self) -> Optional[Dict]:
        """Load training results from JSON"""
        results_path = self.base_dir / 'training_results.json'
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                training_results = json.load(f)
            print(f"✅ Loaded training results from: {results_path}")
            return training_results
        else:
            print(f"⚠️  Training results not found at: {results_path}")
            return None
    
    def apply_validation_fix(self):
        """Apply fix for RT-DETR validation ratio_pad bug"""
        import ultralytics.models.yolo.detect.val as detect_val
        from ultralytics.utils.ops import scale_boxes

        print(f"Start applying RT-DETR validation fix...")
        
        original_scale_preds = detect_val.DetectionValidator.scale_preds
        
        def patched_scale_preds(self, preds, pbatch, orig_imgs):
            """Scale predictions with original image dimensions."""
            predn = preds.clone()
            # Get original dimensions from batch or orig_imgs
            if orig_imgs is not None and len(orig_imgs) > 0:
                ori_shape = orig_imgs[0].shape[:2] if hasattr(orig_imgs[0], 'shape') else preds.shape[-2:]
            else:
                ori_shape = pbatch.get('ori_shape', pbatch.get('resized_shape', preds.shape[-2:]))
            
            # Scale boxes from inference size to original image size
            predn[:, :4] = scale_boxes(
                preds.shape[-2:],  # from shape
                predn[:, :4],  # boxes to scale
                ori_shape,  # to shape
                ratio_pad=pbatch.get('ratio_pad')
            )
            return predn
        
        detect_val.DetectionValidator.scale_preds = patched_scale_preds
        print("✅ Applied RT-DETR validation fix")
    
    def run_validation_pipeline(self,
                                # Model configuration
                                model_path: Optional[str] = None,
                                data_yaml: Optional[str] = None,
                                
                                # Validation configuration
                                batch_size: int = 16,
                                img_size: int = 640,
                                conf_threshold: float = 0.001,
                                iou_threshold: float = 0.6,
                                
                                # Testing options
                                run_test_set: bool = True,
                                test_samples: int = 5,
                                
                                # Tracking options
                                test_video: Optional[str] = None,
                                tracking_conf: float = 0.25,
                                save_video: bool = True) -> Dict:
        """
        Run the validation pipeline: validation -> testing -> tracking
        
        Args:
            model_path: Path to trained model (uses training results if None)
            data_yaml: Path to dataset YAML (uses training results if None)
            batch_size: Validation batch size
            img_size: Image size
            conf_threshold: Confidence threshold for validation
            iou_threshold: IoU threshold for NMS
            run_test_set: Run evaluation on test set
            test_samples: Number of sample images to test
            test_video: Path to test video for tracking
            tracking_conf: Confidence threshold for tracking
            save_video: Save tracked video
            
        Returns:
            Dictionary with validation and tracking results
        """
        
        # ==================== GET MODEL AND DATA PATHS ====================
        if model_path is None:
            if self.training_results and 'rtdetr_training' in self.training_results:
                model_path = self.training_results['rtdetr_training']['model_path']
                print(f"📥 Using model from training: {model_path}")
            else:
                raise ValueError("model_path must be provided or training_results.json must exist")
        
        if data_yaml is None:
            if self.training_results and 'data_preparation' in self.training_results:
                split_dir = self.training_results['data_preparation']['split_dir']
                data_yaml = str(Path(split_dir) / 'dataset.yaml')
                print(f"📥 Using data config: {data_yaml}")
            else:
                raise ValueError("data_yaml must be provided or training_results.json must exist")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
        
        # Apply validation fix
        self.apply_validation_fix()
        
        # ==================== STEP 1: VALIDATION ====================
        print("\n" + "🎯 PHASE 1: MODEL VALIDATION")
        print("="*60)
        
        trainer = RTDETRTrainer()
        
        print("\n🔍 Running validation on validation set...")
        val_results = trainer.validate(
            model_path=model_path,
            data_yaml=data_yaml,
            img_size=img_size,
            batch_size=batch_size,
            conf=conf_threshold,
            iou=iou_threshold
        )
        
        self.results['validation'] = {
            'model_path': model_path,
            'data_yaml': data_yaml,
            'map50': float(val_results.box.map50) if hasattr(val_results, 'box') else None,
            'map50_95': float(val_results.box.map) if hasattr(val_results, 'box') else None,
            'precision': float(val_results.box.mp) if hasattr(val_results, 'box') else None,
            'recall': float(val_results.box.mr) if hasattr(val_results, 'box') else None,
        }
        
        print("\n✅ Validation complete!")
        
        # ==================== STEP 2: TEST SET EVALUATION ====================
        if run_test_set:
            print("\n" + "🎯 PHASE 2: TEST SET EVALUATION")
            print("="*60)
            
            # Load data.yaml to create test config
            import yaml
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Create temporary test yaml
            test_yaml = str(Path(data_yaml).parent / 'test_data_temp.yaml')
            test_config = data_config.copy()
            test_config['val'] = data_config.get('test', data_config['val'])
            
            with open(test_yaml, 'w') as f:
                yaml.dump(test_config, f, default_flow_style=False)
            
            print("\n🧪 Running evaluation on test set...")
            test_results = trainer.validate(
                model_path=model_path,
                data_yaml=test_yaml,
                img_size=img_size,
                batch_size=batch_size,
                conf=conf_threshold,
                iou=iou_threshold
            )
            
            self.results['test'] = {
                'map50': float(test_results.box.map50) if hasattr(test_results, 'box') else None,
                'map50_95': float(test_results.box.map) if hasattr(test_results, 'box') else None,
                'precision': float(test_results.box.mp) if hasattr(test_results, 'box') else None,
                'recall': float(test_results.box.mr) if hasattr(test_results, 'box') else None,
            }
            
            # Clean up temp file
            if os.path.exists(test_yaml):
                os.remove(test_yaml)
            
            print("\n✅ Test evaluation complete!")
        
        # ==================== STEP 3: SAMPLE PREDICTIONS ====================
        print("\n" + "🎯 PHASE 3: SAMPLE PREDICTIONS")
        print("="*60)
        
        print(f"\n🖼️  Testing on {test_samples} sample images...")
        sample_results = trainer.predict(
            source=str(Path(data_yaml).parent / 'val' / 'images'),
            model_path=model_path,
            conf=tracking_conf,
            save=True,
            project=str(self.base_dir / 'runs' / 'predict'),
            name='validation_samples'
        )
        
        self.results['samples'] = {
            'num_samples': len(sample_results) if sample_results else 0,
            'output_dir': str(self.base_dir / 'runs' / 'predict' / 'validation_samples')
        }
        
        print(f"\n✅ Sample predictions saved!")
        
        # ==================== STEP 4: VIDEO TRACKING ====================
        if test_video and Path(test_video).exists():
            print("\n" + "🎯 PHASE 4: VIDEO TRACKING")
            print("="*60)
            
            try:
                # Initialize tracker
                tracker = SAM2Tracker()
                
                # Track video
                print(f"\n🎥 Tracking video: {test_video}")
                tracking_results = tracker.track_video(
                    video_path=test_video,
                    detector_model_path=model_path,
                    output_dir=str(self.base_dir / 'tracking_output'),
                    conf_threshold=tracking_conf,
                    iou_threshold=0.45,
                    save_video=save_video,
                    save_json=True
                )
                
                self.results['tracking'] = {
                    'video_path': test_video,
                    'output_dir': str(self.base_dir / 'tracking_output'),
                    'total_frames': len(tracking_results['frames']),
                    'total_tracks': len(tracking_results['tracks']),
                    'fps': tracking_results['fps']
                }
                
                # Analyze tracks
                print("\n📊 Analyzing tracking results...")
                track_stats = tracker.analyze_tracks(tracking_results)
                self.results['tracking']['statistics'] = track_stats
                
                # Create visualizations
                print("\n🎨 Creating track visualizations...")
                viz_path = tracker.visualize_tracks(
                    tracking_results,
                    output_path=str(self.base_dir / 'tracking_output' / 'track_analysis.png')
                )
                self.results['tracking']['visualization_path'] = viz_path
                
                print("\n✅ Video tracking complete!")
                
            except Exception as e:
                print(f"\n⚠️  Video tracking failed: {e}")
                self.results['tracking']['error'] = str(e)
                
        else:
            if test_video:
                print(f"\n⚠️  Test video not found: {test_video}")
            else:
                print("\n⏩ Skipping video tracking (no test video provided)")
        
        # ==================== SAVE RESULTS ====================
        self._save_results()
        
        # ==================== PRINT SUMMARY ====================
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save validation results to JSON"""
        results_path = self.base_dir / 'validation_results.json'
        
        # Convert any Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        results_json = convert_paths(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n💾 Validation results saved to: {results_path}")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("✨ VALIDATION COMPLETE - SUMMARY")
        print("="*60)
        
        # Validation
        if 'map50' in self.results['validation']:
            print(f"\n📊 Validation Set Results:")
            val = self.results['validation']
            print(f"   mAP50: {val['map50']:.4f}")
            print(f"   mAP50-95: {val['map50_95']:.4f}")
            print(f"   Precision: {val['precision']:.4f}")
            print(f"   Recall: {val['recall']:.4f}")
        
        # Test set
        if 'test' in self.results and 'map50' in self.results['test']:
            print(f"\n📊 Test Set Results:")
            test = self.results['test']
            print(f"   mAP50: {test['map50']:.4f}")
            print(f"   mAP50-95: {test['map50_95']:.4f}")
            print(f"   Precision: {test['precision']:.4f}")
            print(f"   Recall: {test['recall']:.4f}")
        
        # Samples
        if 'num_samples' in self.results['samples']:
            print(f"\n🖼️  Sample Predictions:")
            print(f"   Processed: {self.results['samples']['num_samples']} images")
            print(f"   Output: {self.results['samples']['output_dir']}")
        
        # Tracking
        if 'total_tracks' in self.results['tracking']:
            print(f"\n🎬 Video Tracking:")
            print(f"   Video: {self.results['tracking']['video_path']}")
            print(f"   Frames: {self.results['tracking']['total_frames']}")
            print(f"   Tracks: {self.results['tracking']['total_tracks']}")
            print(f"   Output: {self.results['tracking']['output_dir']}")
        
        print("\n" + "="*60)
        print(f"📁 All outputs saved to: {self.base_dir}")
        print("="*60)


def main():
    """
    Example usage of validation pipeline
    """
    
    # Initialize pipeline
    pipeline = BasketballValidationPipeline(base_dir="./basketball_pipeline")
    
    # Option 1: Run with automatic model/data detection from training
    results = pipeline.run_validation_pipeline(
        # Model and data (auto-detected from training_results.json if None)
        model_path=None,  # or specify: "./basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt"
        data_yaml=None,   # or specify: "./basketball_pipeline/data/split_dataset/dataset.yaml"
        
        # Validation configuration
        batch_size=16,
        img_size=640,
        conf_threshold=0.001,
        iou_threshold=0.6,
        
        # Testing options
        run_test_set=True,
        test_samples=5,
        
        # Tracking (optional - provide path to test video)
        test_video=None,  # e.g., "./test_videos/game1.mp4"
        tracking_conf=0.25,
        save_video=True
    )
    
    # Option 2: Run with explicit paths
    # results = pipeline.run_validation_pipeline(
    #     model_path="./basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt",
    #     data_yaml="./basketball_pipeline/data/split_dataset/dataset.yaml",
    #     test_video="./test_videos/game1.mp4",
    #     batch_size=16,
    #     run_test_set=True
    # )
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("✅ VALIDATION PIPELINE COMPLETE")
    print("="*60)
    print("\n🎉 All done! Check the output directory for results:")
    print(f"   - Validation metrics: ./basketball_pipeline/validation_results.json")
    print(f"   - Sample predictions: ./basketball_pipeline/runs/predict/")
    print(f"   - Tracking output: ./basketball_pipeline/tracking_output/")
