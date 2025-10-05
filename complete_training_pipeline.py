"""
Complete Basketball Player Detection and Tracking Pipeline
Integrates: Data preparation -> RT-DETR training -> SAM2 tracking
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime



# Import your existing modules
# Assumes the following structure:
# - train.py (your existing main training script)
# - rtdetr_training.py (RT-DETR trainer)
# - sam2_tracker.py (SAM2 tracker)

try:
    from train import TrainingPipeline
    from rtdetr_training import RTDETRTrainer
    from sam2_tracker import SAM2Tracker
except ImportError:
    print("⚠️  Warning: Could not import all modules. Make sure files are in the same directory.")


class BasketballTrainingPipeline:
    """
    Complete end-to-end pipeline for basketball player detection and tracking
    """
    
    def __init__(self, base_dir: str = "./basketball_pipeline"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.data_pipeline = TrainingPipeline(str(self.base_dir / "data"))
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'data_preparation': {},
            'rtdetr_training': {},
            'tracking': {},
        }
        
        print("🏀 Basketball Player Detection & Tracking Pipeline")
        print("="*60)
        print(f"Working directory: {self.base_dir}")
        print("="*60)
    
    def run_full_pipeline(self,
                         # Data preparation args
                         download_data: bool = True,
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.2,
                         
                         # RT-DETR training args
                         model_size: str = 'rtdetr-l',
                         epochs: int = 100,
                         batch_size: int = 16,
                         img_size: int = 640,
                         pretrained: bool = True,
                         
                         # Tracking args
                         test_video: Optional[str] = None,
                         conf_threshold: float = 0.25,
                         
                         # General args
                         skip_training: bool = False,
                         model_path: Optional[str] = None) -> Dict:
        """
        Run the complete pipeline from data to tracking
        
        Args:
            download_data: Whether to download data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            model_size: RT-DETR model size
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            pretrained: Use pretrained weights
            test_video: Path to test video for tracking
            conf_threshold: Detection confidence threshold
            skip_training: Skip training and use existing model
            model_path: Path to existing model (if skip_training=True)
            
        Returns:
            Dictionary with all results
        """
        
        # ==================== STEP 1: DATA PREPARATION ====================
        if not skip_training:
            print("\n" + "🎯 PHASE 1: DATA PREPARATION")
            print("="*60)
            
            data_results = self.data_pipeline.run_pipeline(
                download=download_data,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )
            
            self.results['data_preparation'] = {
                'dataset_path': data_results['dataset_path'],
                'split_dir': data_results['split_stats']['output_dir'],
                'is_valid': data_results['is_valid_coco'],
                'split_stats': data_results['split_stats']
            }
            
            split_dir = data_results['split_stats']['output_dir']
        else:
            # Assume data is already prepared
            split_dir = str(self.base_dir / "data" / "split_dataset")
            print(f"\n⏩ Skipping data preparation, using: {split_dir}")
            self.results['data_preparation']['split_dir'] = split_dir
        
        # ==================== STEP 2: RT-DETR TRAINING ====================
        if not skip_training:
            print("\n" + "🎯 PHASE 2: RT-DETR MODEL TRAINING")
            print("="*60)
            
            # Initialize trainer
            trainer = RTDETRTrainer(model_size=model_size)
            
            # Prepare dataset YAML
            data_yaml = trainer.prepare_dataset_yaml(split_dir)
            
            # Train model
            print(f"\n🚀 Starting training with {epochs} epochs...")
            training_results = trainer.train(
                data_yaml=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                project=str(self.base_dir / 'runs' / 'train'),
                name='basketball_rtdetr',
                pretrained=pretrained,
                augment=True,
                patience=50,
                save=True,
                plots=True
            )
            
            self.results['rtdetr_training'] = training_results
            best_model_path = training_results['model_path']
            
            # Validate model
            print("\n📊 Validating trained model...")
            val_results = trainer.validate(
                model_path=best_model_path,
                data_yaml=data_yaml,
                img_size=img_size
            )
            
            self.results['rtdetr_training']['validation'] = {
                'map50': float(val_results.box.map50) if hasattr(val_results, 'box') else None,
                'map50_95': float(val_results.box.map) if hasattr(val_results, 'box') else None,
                'precision': float(val_results.box.mp) if hasattr(val_results, 'box') else None,
                'recall': float(val_results.box.mr) if hasattr(val_results, 'box') else None,
            }
            
            # Export model to ONNX for deployment
            print("\n📦 Exporting model to ONNX...")
            try:
                onnx_path = trainer.export_model(
                    model_path=best_model_path,
                    format='onnx',
                    img_size=img_size
                )
                self.results['rtdetr_training']['onnx_path'] = onnx_path
            except Exception as e:
                print(f"⚠️  ONNX export failed: {e}")
                self.results['rtdetr_training']['onnx_path'] = None
            
        else:
            # Use provided model
            if model_path is None:
                raise ValueError("model_path must be provided when skip_training=True")
            
            print(f"\n⏩ Skipping training, using existing model: {model_path}")
            best_model_path = model_path
            self.results['rtdetr_training']['model_path'] = model_path
        
        # ==================== STEP 3: VIDEO TRACKING ====================
        if test_video and Path(test_video).exists():
            print("\n" + "🎯 PHASE 3: VIDEO TRACKING")
            print("="*60)
            
            # Initialize tracker
            tracker = SAM2Tracker()
            
            # Track video
            print(f"\n🎥 Tracking video: {test_video}")
            tracking_results = tracker.track_video(
                video_path=test_video,
                detector_model_path=best_model_path,
                output_dir=str(self.base_dir / 'tracking_output'),
                conf_threshold=conf_threshold,
                iou_threshold=0.45,
                save_video=True,
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
            
        else:
            if test_video:
                print(f"\n⚠️  Test video not found: {test_video}")
            print("⏩ Skipping tracking phase (no test video provided)")
        
        # ==================== SAVE FINAL RESULTS ====================
        self._save_results()
        
        # ==================== PRINT SUMMARY ====================
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save pipeline results to JSON"""
        results_path = self.base_dir / 'pipeline_results.json'
        
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
        
        print(f"\n💾 Results saved to: {results_path}")
    
    def _print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("✨ PIPELINE COMPLETE - SUMMARY")
        print("="*60)
        
        # Data preparation
        if 'split_dir' in self.results['data_preparation']:
            print(f"\n📊 Data Preparation:")
            print(f"   Split directory: {self.results['data_preparation']['split_dir']}")
            if 'split_stats' in self.results['data_preparation']:
                splits = self.results['data_preparation']['split_stats']['splits']
                for split_name, stats in splits.items():
                    print(f"   {split_name.upper()}: {stats['num_images']} images, "
                          f"{stats['num_annotations']} annotations")
        
        # Training
        if 'model_path' in self.results['rtdetr_training']:
            print(f"\n🏋️ Model Training:")
            print(f"   Model: {self.results['rtdetr_training']['model_path']}")
            if 'validation' in self.results['rtdetr_training']:
                val = self.results['rtdetr_training']['validation']
                if val.get('map50'):
                    print(f"   mAP50: {val['map50']:.4f}")
                    print(f"   mAP50-95: {val['map50_95']:.4f}")
                    print(f"   Precision: {val['precision']:.4f}")
                    print(f"   Recall: {val['recall']:.4f}")
        
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
    Example usage of complete pipeline
    """

 
    # Initialize pipeline
    pipeline = BasketballTrainingPipeline(base_dir="./basketball_pipeline")
    
    # Option 1: Run full pipeline (download -> train -> track)
    results = pipeline.run_full_pipeline(
        # Data
        download_data=True,
        train_ratio=0.8,
        val_ratio=0.2,
        
        # Training
        model_size='rtdetr-l',  # or 'rtdetr-x' for larger model
        epochs=20,
        batch_size=8,
        img_size=640,
        pretrained=True,
        
        # Tracking (optional - provide path to test video)
        test_video=None,  # e.g., "./test_videos/game1.mp4"
        conf_threshold=0.25,
        
        # Control
        skip_training=False,
        model_path=None,


        lr0=0.001,  # LOWER learning rate (10x smaller)
        optimizer='SGD',  # MORE STABLE than AdamW
        amp=False,  # DISABLE mixed precision
        deterministic=False,  # DISABLE deterministic mode
        patience=20,  # Reduce patience
    )
    
    # Option 2: Skip training and just run tracking with existing model
    # results = pipeline.run_full_pipeline(
    #     skip_training=True,
    #     model_path="./basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt",
    #     test_video="./test_videos/game1.mp4",
    #     conf_threshold=0.25
    # )
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n🎉 All done! Check the output directory for results.")
    print("\nNext steps:")
    print("1. Review training metrics in: ./basketball_pipeline/runs/train/")
    print("2. Check tracked videos in: ./basketball_pipeline/tracking_output/")
    print("3. Fine-tune hyperparameters if needed")
    print("4. Deploy model for real-time inference")