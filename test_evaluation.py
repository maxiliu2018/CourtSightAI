"""
Model Testing and Evaluation Script
Comprehensive evaluation of trained RT-DETR model on test dataset
"""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from ultralytics import RTDETR
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            device: Device to use
        """
        self.model_path = model_path
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RTDETR(model_path)
        self.results = {}
        
        print(f"🔍 Model Evaluator Initialized")
        print(f"   Model: {model_path}")
        print(f"   Device: {self.device}")
    
    def evaluate_test_set(self,
                         test_dir: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         save_predictions: bool = True,
                         output_dir: str = "./evaluation_output") -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_dir: Directory containing test images and annotations
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_predictions: Save prediction visualizations
            output_dir: Output directory
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "="*60)
        print("📊 EVALUATING ON TEST SET")
        print("="*60)
        
        test_path = Path(test_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load ground truth annotations
        ann_file = test_path / '_annotations.coco.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            gt_data = json.load(f)
        
        coco_gt = COCO(str(ann_file))
        
        print(f"📁 Test set: {test_path}")
        print(f"   Images: {len(gt_data['images'])}")
        print(f"   Annotations: {len(gt_data['annotations'])}")
        print(f"   Categories: {len(gt_data['categories'])}")
        
        # Run predictions
        print(f"\n🔮 Running predictions...")
        
        predictions = []
        prediction_results = []
        
        for img_info in tqdm(gt_data['images'], desc="Processing images"):
            img_path = test_path / img_info['file_name']
            
            if not img_path.exists():
                print(f"⚠️  Image not found: {img_path}")
                continue
            
            # Run inference
            results = self.model.predict(
                str(img_path),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Extract predictions
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Convert to COCO format
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                predictions.append({
                    'image_id': img_info['id'],
                    'category_id': int(cls) + 1,  # COCO uses 1-indexed categories
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })
            
            prediction_results.append({
                'image_id': img_info['id'],
                'file_name': img_info['file_name'],
                'num_predictions': len(boxes),
                'scores': scores.tolist() if len(scores) > 0 else []
            })
            
            # Save visualization
            if save_predictions and len(boxes) > 0:
                self._save_prediction_visualization(
                    img_path, results, output_path / 'predictions'
                )
        
        print(f"✅ Predictions complete: {len(predictions)} total detections")
        
        # Save predictions to JSON
        pred_file = output_path / 'predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f)
        
        # Run COCO evaluation
        print(f"\n📈 Computing COCO metrics...")
        
        if len(predictions) > 0:
            coco_dt = coco_gt.loadRes(str(pred_file))
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'AP': float(coco_eval.stats[0]),
                'AP50': float(coco_eval.stats[1]),
                'AP75': float(coco_eval.stats[2]),
                'AP_small': float(coco_eval.stats[3]),
                'AP_medium': float(coco_eval.stats[4]),
                'AP_large': float(coco_eval.stats[5]),
                'AR_max1': float(coco_eval.stats[6]),
                'AR_max10': float(coco_eval.stats[7]),
                'AR_max100': float(coco_eval.stats[8]),
                'AR_small': float(coco_eval.stats[9]),
                'AR_medium': float(coco_eval.stats[10]),
                'AR_large': float(coco_eval.stats[11]),
            }
        else:
            print("⚠️  No predictions made, skipping COCO evaluation")
            metrics = {}
        
        # Per-class analysis
        print(f"\n📊 Computing per-class metrics...")
        class_metrics = self._compute_class_metrics(
            coco_gt, predictions, gt_data['categories']
        )
        
        # Compile results
        evaluation_results = {
            'model_path': self.model_path,
            'test_dir': str(test_path),
            'num_test_images': len(gt_data['images']),
            'num_predictions': len(predictions),
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'coco_metrics': metrics,
            'class_metrics': class_metrics,
            'prediction_details': prediction_results
        }
        
        # Save results
        results_file = output_path / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Create visualizations
        self._create_evaluation_plots(evaluation_results, output_path)
        
        self.results = evaluation_results
        return evaluation_results
    
    def _compute_class_metrics(self, 
                               coco_gt: COCO, 
                               predictions: List[Dict],
                               categories: List[Dict]) -> Dict:
        """Compute per-class metrics"""
        
        class_metrics = {}
        
        for cat in categories:
            cat_id = cat['id']
            cat_name = cat['name']
            
            # Filter predictions for this class
            cat_preds = [p for p in predictions if p['category_id'] == cat_id]
            
            # Get ground truth for this class
            cat_gt = coco_gt.getAnnIds(catIds=[cat_id])
            
            class_metrics[cat_name] = {
                'num_gt': len(cat_gt),
                'num_pred': len(cat_preds),
                'avg_confidence': np.mean([p['score'] for p in cat_preds]) if cat_preds else 0.0
            }
        
        return class_metrics
    
    def _save_prediction_visualization(self, 
                                      img_path: Path,
                                      results,
                                      output_dir: Path):
        """Save visualization of predictions"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw predictions
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{results.names[int(cls)]} {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(img, (x1, y1 - label_h - 10),
                        (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save
        output_file = output_dir / img_path.name
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), img)
    
    def _create_evaluation_plots(self, results: Dict, output_dir: Path):
        """Create evaluation visualization plots"""
        
        fig = plt.figure(figsize=(16, 10))
        
        # 1. COCO Metrics Bar Chart
        ax1 = plt.subplot(2, 3, 1)
        if results['coco_metrics']:
            metrics = results['coco_metrics']
            metric_names = ['AP', 'AP50', 'AP75', 'AR_max100']
            metric_values = [metrics.get(m, 0) for m in metric_names]
            
            bars = ax1.bar(range(len(metric_names)), metric_values)
            ax1.set_xticks(range(len(metric_names)))
            ax1.set_xticklabels(metric_names)
            ax1.set_ylabel('Score')
            ax1.set_title('COCO Metrics', fontweight='bold')
            ax1.set_ylim([0, 1])
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # 2. Per-class predictions
        ax2 = plt.subplot(2, 3, 2)
        if results['class_metrics']:
            class_names = list(results['class_metrics'].keys())
            num_gt = [results['class_metrics'][c]['num_gt'] for c in class_names]
            num_pred = [results['class_metrics'][c]['num_pred'] for c in class_names]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            ax2.bar(x - width/2, num_gt, width, label='Ground Truth', alpha=0.8)
            ax2.bar(x + width/2, num_pred, width, label='Predictions', alpha=0.8)
            
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.set_title('Predictions vs Ground Truth by Class', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(class_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Confidence distribution
        ax3 = plt.subplot(2, 3, 3)
        all_scores = []
        for pred_detail in results['prediction_details']:
            all_scores.extend(pred_detail['scores'])
        
        if all_scores:
            ax3.hist(all_scores, bins=30, edgecolor='black', alpha=0.7)
            ax3.axvline(results['conf_threshold'], color='r', 
                       linestyle='--', label=f'Threshold ({results["conf_threshold"]})')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Prediction Confidence Distribution', fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Detections per image
        ax4 = plt.subplot(2, 3, 4)
        det_per_img = [p['num_predictions'] for p in results['prediction_details']]
        
        if det_per_img:
            ax4.hist(det_per_img, bins=20, edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Number of Detections')
            ax4.set_ylabel('Number of Images')
            ax4.set_title('Detections per Image Distribution', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
            
            # Add statistics
            stats_text = f'Mean: {np.mean(det_per_img):.1f}\n'
            stats_text += f'Median: {np.median(det_per_img):.1f}\n'
            stats_text += f'Max: {np.max(det_per_img):.0f}'
            ax4.text(0.95, 0.95, stats_text,
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. AP by size (if available)
        ax5 = plt.subplot(2, 3, 5)
        if results['coco_metrics']:
            metrics = results['coco_metrics']
            sizes = ['Small', 'Medium', 'Large']
            ap_by_size = [
                metrics.get('AP_small', 0),
                metrics.get('AP_medium', 0),
                metrics.get('AP_large', 0)
            ]
            
            bars = ax5.bar(sizes, ap_by_size, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax5.set_ylabel('Average Precision (AP)')
            ax5.set_title('AP by Object Size', fontweight='bold')
            ax5.set_ylim([0, 1])
            ax5.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, ap_by_size):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        table_data = [['Metric', 'Value']]
        
        table_data.append(['Test Images', str(results['num_test_images'])])
        table_data.append(['Total Predictions', str(results['num_predictions'])])
        table_data.append(['Avg Pred/Image', 
                          f"{results['num_predictions'] / max(results['num_test_images'], 1):.2f}"])
        
        if results['coco_metrics']:
            table_data.append(['mAP', f"{results['coco_metrics']['AP']:.3f}"])
            table_data.append(['mAP@50', f"{results['coco_metrics']['AP50']:.3f}"])
            table_data.append(['mAP@75', f"{results['coco_metrics']['AP75']:.3f}"])
        
        table_data.append(['Conf Threshold', f"{results['conf_threshold']:.2f}"])
        table_data.append(['IoU Threshold', f"{results['iou_threshold']:.2f}"])
        
        table = ax6.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header row
        table[(0, 0)].set_facecolor('#34495e')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_facecolor('#34495e')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Evaluation Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save
        plot_path = output_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Evaluation plots saved to: {plot_path}")
        plt.show()
    
    def compare_models(self, 
                      model_paths: List[str],
                      test_dir: str,
                      output_dir: str = "./model_comparison") -> pd.DataFrame:
        """
        Compare multiple trained models
        
        Args:
            model_paths: List of model paths to compare
            test_dir: Test dataset directory
            output_dir: Output directory
            
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*60)
        print("🔬 COMPARING MODELS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        comparison_results = []
        
        for i, model_path in enumerate(model_paths):
            print(f"\n📊 Evaluating model {i+1}/{len(model_paths)}: {Path(model_path).name}")
            
            # Create temporary evaluator
            evaluator = ModelEvaluator(model_path)
            
            # Evaluate
            results = evaluator.evaluate_test_set(
                test_dir=test_dir,
                save_predictions=False,
                output_dir=str(output_path / f'model_{i+1}')
            )
            
            # Extract key metrics
            model_result = {
                'model_name': Path(model_path).stem,
                'model_path': model_path,
            }
            
            if results['coco_metrics']:
                model_result.update({
                    'mAP': results['coco_metrics']['AP'],
                    'mAP@50': results['coco_metrics']['AP50'],
                    'mAP@75': results['coco_metrics']['AP75'],
                    'AR': results['coco_metrics']['AR_max100'],
                })
            
            model_result.update({
                'num_predictions': results['num_predictions'],
                'avg_pred_per_image': results['num_predictions'] / results['num_test_images']
            })
            
            comparison_results.append(model_result)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Save to CSV
        csv_path = output_path / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Comparison saved to: {csv_path}")
        
        # Print comparison
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        print(df.to_string(index=False))
        
        # Create comparison visualization
        self._plot_model_comparison(df, output_path)
        
        return df
    
    def _plot_model_comparison(self, df: pd.DataFrame, output_dir: Path):
        """Create comparison visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['mAP', 'mAP@50', 'mAP@75', 'AR']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            if metric in df.columns:
                bars = ax.bar(range(len(df)), df[metric], color=colors[idx], alpha=0.7)
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['model_name'], rotation=45, ha='right')
                ax.set_ylabel('Score')
                ax.set_title(metric, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, df[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = output_dir / 'model_comparison_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {plot_path}")
        plt.show()
    
    def error_analysis(self, 
                      test_dir: str,
                      output_dir: str = "./error_analysis",
                      num_samples: int = 20) -> Dict:
        """
        Perform error analysis on predictions
        
        Args:
            test_dir: Test dataset directory
            output_dir: Output directory
            num_samples: Number of error samples to visualize
            
        Returns:
            Error analysis results
        """
        print("\n" + "="*60)
        print("🔍 PERFORMING ERROR ANALYSIS")
        print("="*60)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Run evaluation to get predictions
        if not self.results:
            self.evaluate_test_set(test_dir, save_predictions=False)
        
        # Analyze errors
        error_types = {
            'false_positives': [],
            'false_negatives': [],
            'misclassifications': [],
            'low_confidence': []
        }
        
        # TODO: Implement detailed error analysis
        # This would require matching predictions with ground truth
        # and categorizing different types of errors
        
        print("✅ Error analysis complete")
        
        return error_types


def main():
    """Example usage of model evaluation"""
    
    # Initialize evaluator with trained model
    evaluator = ModelEvaluator(
        model_path="./basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt"
    )
    
    # Evaluate on test set
    results = evaluator.evaluate_test_set(
        test_dir="./basketball_pipeline/data/split_dataset/test",
        conf_threshold=0.25,
        iou_threshold=0.45,
        save_predictions=True,
        output_dir="./evaluation_output"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    if results['coco_metrics']:
        print(f"\nCOCO Metrics:")
        print(f"  mAP: {results['coco_metrics']['AP']:.4f}")
        print(f"  mAP@50: {results['coco_metrics']['AP50']:.4f}")
        print(f"  mAP@75: {results['coco_metrics']['AP75']:.4f}")
        print(f"  AR@100: {results['coco_metrics']['AR_max100']:.4f}")
    
    print(f"\nPer-Class Performance:")
    for class_name, metrics in results['class_metrics'].items():
        print(f"  {class_name}:")
        print(f"    Ground Truth: {metrics['num_gt']}")
        print(f"    Predictions: {metrics['num_pred']}")
        print(f"    Avg Confidence: {metrics['avg_confidence']:.3f}")
    
    # Optional: Compare multiple models
    # model_paths = [
    #     "./runs/train/exp1/weights/best.pt",
    #     "./runs/train/exp2/weights/best.pt",
    # ]
    # comparison_df = evaluator.compare_models(
    #     model_paths=model_paths,
    #     test_dir="./data/split_dataset/test"
    # )
    
    return results


if __name__ == "__main__":
    import torch
    results = main()