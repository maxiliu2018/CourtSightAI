"""
Basketball Player Detection Training Pipeline
Part 1: Data preparation and model training only
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

# Import your existing modules
try:
    from train import TrainingPipeline
    from rtdetr_training import RTDETRTrainer
except ImportError:
    print("⚠️  Warning: Could not import all modules. Make sure files are in the same directory.")


class BasketballTrainingPipeline:
    """
    Training pipeline for basketball player detection
    Handles data preparation and RT-DETR model training
    """
    
    def __init__(self, base_dir: str = "./basketball_pipeline"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.data_pipeline = TrainingPipeline(str(self.base_dir / "data"))
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'data_preparation': {},
            'rtdetr_training': {},
        }
        
        print("🏀 Basketball Player Detection Training Pipeline")
        print("="*60)
        print(f"Working directory: {self.base_dir}")
        print("="*60)
    
    def run_training_pipeline(self,
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
                             lr0: float = 0.001,
                             optimizer: str = 'AdamW',
                             patience: int = 50,
                             
                             # Export options
                             export_onnx: bool = True) -> Dict:
        """
        Run the training pipeline: data preparation -> model training
        
        Args:
            download_data: Whether to download data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            model_size: RT-DETR model size ('rtdetr-l', 'rtdetr-x')
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            pretrained: Use pretrained weights
            lr0: Initial learning rate
            optimizer: Optimizer type ('AdamW', 'SGD', etc.)
            patience: Early stopping patience
            export_onnx: Export model to ONNX format
            
        Returns:
            Dictionary with training results
        """
        
        # ==================== STEP 1: DATA PREPARATION ====================
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
        
        print("\n✅ Data preparation complete!")
        print(f"   Split directory: {split_dir}")
        
        # ==================== STEP 2: RT-DETR TRAINING ====================
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
            patience=patience,
            save=True,
            plots=True,
            lr0=lr0,
            optimizer=optimizer
        )
        
        self.results['rtdetr_training'] = training_results
        best_model_path = training_results['model_path']
        
        print("\n✅ Training complete!")
        print(f"   Best model: {best_model_path}")
        
        # ==================== STEP 3: EXPORT MODEL ====================
        if export_onnx:
            print("\n📦 Exporting model to ONNX...")
            try:
                onnx_path = trainer.export_model(
                    model_path=best_model_path,
                    format='onnx',
                    img_size=img_size
                )
                self.results['rtdetr_training']['onnx_path'] = onnx_path
                print(f"✅ ONNX model saved: {onnx_path}")
            except Exception as e:
                print(f"⚠️  ONNX export failed: {e}")
                self.results['rtdetr_training']['onnx_path'] = None
        
        # ==================== SAVE TRAINING INFO ====================
        self._save_results()
        
        # ==================== PRINT SUMMARY ====================
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save pipeline results to JSON"""
        results_path = self.base_dir / 'training_results.json'
        
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
        
        print(f"\n💾 Training results saved to: {results_path}")
    
    def _print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("✨ TRAINING COMPLETE - SUMMARY")
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
            print(f"   Best model: {self.results['rtdetr_training']['model_path']}")
            print(f"   Last model: {self.results['rtdetr_training'].get('last_model_path', 'N/A')}")
            print(f"   Results dir: {self.results['rtdetr_training']['results_dir']}")
            
            if self.results['rtdetr_training'].get('onnx_path'):
                print(f"   ONNX export: {self.results['rtdetr_training']['onnx_path']}")
        
        print("\n" + "="*60)
        print(f"📁 All outputs saved to: {self.base_dir}")
        print("="*60)
        print("\n💡 Next steps:")
        print("   1. Run pipeline_validation.py to validate your model")
        print("   2. Check training plots in the results directory")
        print("   3. Fine-tune hyperparameters if needed")


def main():
    """
    Example usage of training pipeline
    """
    
    # Initialize pipeline
    pipeline = BasketballTrainingPipeline(base_dir="./basketball_pipeline")
    
    # Run training pipeline
    results = pipeline.run_training_pipeline(
        # Data preparation
        download_data=True,
        train_ratio=0.8,
        val_ratio=0.2,
        
        # Training configuration
        model_size='rtdetr-l',  # or 'rtdetr-x' for larger model
        epochs=20,
        batch_size=8,
        img_size=640,
        pretrained=True,
        lr0=0.001,
        optimizer='AdamW',
        patience=50,
        
        # Export options
        export_onnx=True
    )
    
    print("\n🎉 Training complete!")
    print(f"\n📁 Model saved at: {results['rtdetr_training']['model_path']}")
    print(f"📊 Results saved at: {results['rtdetr_training']['results_dir']}")
    print(f"📝 Training info: {pipeline.base_dir}/training_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*60)
    print("\nNext: Run pipeline_validation.py to validate and test your model!")
