"""
Main training script that orchestrates data download, verification, splitting, and visualization
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict


# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.verify_coco import verify_coco_dataset
from src.split_data import split_coco_dataset


class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
    def download_data(self) -> str:
        """Run download script and return dataset path"""
        print("\n" + "="*50)
        print("Step 1: Downloading Dataset")
        print("="*50)
        
        # Run download script
        download_script = Path(__file__).parent / "download.py"
        print(download_script)
        result = subprocess.run([sys.executable, str(download_script)], 
                              capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)
        
        # Parse output to get dataset location
        for line in result.stdout.split('\n'):
            if "Saved to:" in line:
                dataset_path = line.split("Saved to:")[-1].strip()
                print(f"✅ Dataset downloaded to: {dataset_path}")
                return dataset_path
                
        # If parsing fails, try to find the dataset
        # Roboflow typically saves to a pattern like this
        possible_paths = list(Path.cwd().glob("**/basketball-bs0zc-*"))
        if possible_paths:
            dataset_path = str(possible_paths[0])
            print(f"✅ Dataset found at: {dataset_path}")
            return dataset_path
            
        raise RuntimeError("Could not determine dataset location after download")
    
    def verify_dataset(self, dataset_path: str) -> bool:
        """Verify COCO format of dataset"""
        print("\n" + "="*50)
        print("Step 2: Verifying COCO Format")
        print("="*50)
        
        is_valid, report = verify_coco_dataset(dataset_path)
        
        print(f"Dataset path: {dataset_path}")
        print(f"Format valid: {'✅ Yes' if is_valid else '❌ No'}")
        
        if report['errors']:
            print(f"\nErrors found ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"  ❌ {error}")
                
        if report['warnings']:
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"  ⚠️  {warning}")
                
        if report['statistics']:
            print(f"\nDataset Statistics:")
            for split, stats in report['statistics'].items():
                print(f"\n  {split.upper()}:")
                print(f"    Images: {stats['num_images']}")
                print(f"    Annotations: {stats['num_annotations']}")
                print(f"    Categories: {stats['num_categories']}")
                if stats.get('categories'):
                    print(f"    Classes: {', '.join(stats['categories'][:5])}")
                    if len(stats['categories']) > 5:
                        print(f"             ... and {len(stats['categories']) - 5} more")
                        
        return is_valid
    
    def split_dataset(self, dataset_path: str, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.2,
                     output_dir: Optional[str] = None) -> Dict:
        """Split dataset into train/val sets"""
        print("\n" + "="*50)
        print("Step 3: Splitting Dataset")
        print("="*50)
        print(f"Split ratio: {train_ratio:.0%} train / {val_ratio:.0%} validation")
        
        if output_dir is None:
            output_dir = str(self.base_dir / "split_dataset")
            
        stats = split_coco_dataset(
            dataset_path=dataset_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=0.0,
            output_dir=output_dir,
            stratify=True,
            seed=42
        )
        
        print(f"✅ Dataset split complete")
        print(f"Output directory: {stats['output_dir']}")
        
        for split_name, split_stats in stats['splits'].items():
            print(f"\n{split_name.upper()} Set:")
            print(f"  Images: {split_stats['num_images']}")
            print(f"  Annotations: {split_stats['num_annotations']}")
            
        return stats
    
    def visualize_split(self, split_stats: Dict, save_path: Optional[str] = None):
        """Create visualization plots for the data split"""
        print("\n" + "="*50)
        print("Step 4: Creating Visualizations")
        print("="*50)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Pie chart of data split
        ax1 = plt.subplot(2, 3, 1)
        splits = list(split_stats['splits'].keys())
        sizes = [split_stats['splits'][s]['num_images'] for s in splits]
        colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(splits)]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=[s.upper() for s in splits], 
                                           colors=colors, autopct='%1.1f%%',
                                           startangle=90)
        ax1.set_title('Dataset Split Distribution\n(by number of images)', fontsize=12, fontweight='bold')
        
        # 2. Bar chart of images per split
        ax2 = plt.subplot(2, 3, 2)
        x_pos = np.arange(len(splits))
        bars = ax2.bar(x_pos, sizes, color=colors)
        ax2.set_xlabel('Split')
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Images per Split', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.upper() for s in splits])
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(size)}', ha='center', va='bottom')
        
        # 3. Annotations per split
        ax3 = plt.subplot(2, 3, 3)
        ann_counts = [split_stats['splits'][s]['num_annotations'] for s in splits]
        bars = ax3.bar(x_pos, ann_counts, color=colors)
        ax3.set_xlabel('Split')
        ax3.set_ylabel('Number of Annotations')
        ax3.set_title('Annotations per Split', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([s.upper() for s in splits])
        
        for bar, count in zip(bars, ann_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom')
        
        # 4. Category distribution across splits
        ax4 = plt.subplot(2, 3, 4)
        
        # Collect category distributions
        all_categories = set()
        for split_data in split_stats['splits'].values():
            if 'category_distribution' in split_data:
                all_categories.update(split_data['category_distribution'].keys())
        
        all_categories = sorted(list(all_categories))
        
        if all_categories:
            # Create grouped bar chart
            n_cats = len(all_categories)
            n_splits = len(splits)
            bar_width = 0.8 / n_splits
            x_pos = np.arange(n_cats)
            
            for i, split in enumerate(splits):
                cat_dist = split_stats['splits'][split].get('category_distribution', {})
                values = [cat_dist.get(cat, 0) for cat in all_categories]
                offset = (i - n_splits/2 + 0.5) * bar_width
                ax4.bar(x_pos + offset, values, bar_width, 
                       label=split.upper(), color=colors[i])
            
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Number of Annotations')
            ax4.set_title('Category Distribution Across Splits', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(all_categories, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
        
        # 5. Annotations per image statistics
        ax5 = plt.subplot(2, 3, 5)
        
        ann_per_img = []
        labels = []
        for split in splits:
            n_imgs = split_stats['splits'][split]['num_images']
            n_anns = split_stats['splits'][split]['num_annotations']
            if n_imgs > 0:
                avg_ann = n_anns / n_imgs
                ann_per_img.append(avg_ann)
                labels.append(split.upper())
        
        bars = ax5.bar(range(len(labels)), ann_per_img, color=colors[:len(labels)])
        ax5.set_xlabel('Split')
        ax5.set_ylabel('Average Annotations per Image')
        ax5.set_title('Average Annotations per Image', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels)
        
        for bar, val in zip(bars, ann_per_img):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        # 6. Summary statistics table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        table_data = [['Metric', 'Train', 'Val', 'Total']]
        
        train_stats = split_stats['splits'].get('train', {})
        val_stats = split_stats['splits'].get('val', {})
        
        metrics = [
            ('Images', 'num_images'),
            ('Annotations', 'num_annotations'),
            ('Categories', 'num_categories')
        ]
        
        total_imgs = 0
        total_anns = 0
        
        for metric_name, metric_key in metrics:
            train_val = train_stats.get(metric_key, 0)
            val_val = val_stats.get(metric_key, 0)
            
            if metric_key == 'num_categories':
                # Categories should be the same across splits
                total_val = train_val
            else:
                total_val = train_val + val_val
                if metric_key == 'num_images':
                    total_imgs = total_val
                elif metric_key == 'num_annotations':
                    total_anns = total_val
                    
            table_data.append([metric_name, str(train_val), str(val_val), str(total_val)])
        
        # Add average annotations per image
        train_avg = f"{train_stats.get('num_annotations', 0) / max(train_stats.get('num_images', 1), 1):.2f}"
        val_avg = f"{val_stats.get('num_annotations', 0) / max(val_stats.get('num_images', 1), 1):.2f}"
        total_avg = f"{total_anns / max(total_imgs, 1):.2f}"
        table_data.append(['Avg Ann/Img', train_avg, val_avg, total_avg])
        
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('COCO Dataset Split Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        else:
            save_path = "dataset_split_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
            
        plt.show()
        
        return save_path
    
    def run_pipeline(self, 
                    download: bool = True,
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.2) -> Dict:
        """Run complete training pipeline"""
        
        print("\n" + "🚀 Starting Training Pipeline " + "🚀")
        print("="*50)
        
        results = {}
        
        try:
            # Step 1: Download data
            if download:
                dataset_path = self.download_data()
            else:
                # Find existing dataset
                possible_paths = list(Path.cwd().glob("**/basketball-bs0zc-*"))
                if not possible_paths:
                    raise ValueError("No dataset found. Please run with download=True")
                dataset_path = str(possible_paths[0])
                print(f"Using existing dataset: {dataset_path}")
                
            results['dataset_path'] = dataset_path
            
            # Step 2: Verify COCO format
            is_valid = self.verify_dataset(dataset_path)
            results['is_valid_coco'] = is_valid
            
            if not is_valid:
                print("\n⚠️  Dataset has COCO format issues but continuing anyway...")
            
            # Step 3: Split dataset
            split_stats = self.split_dataset(dataset_path, train_ratio, val_ratio)
            results['split_stats'] = split_stats
            
            # Step 4: Create visualizations
            viz_path = self.visualize_split(split_stats)
            results['visualization_path'] = viz_path
            
            print("\n" + "="*50)
            print("✨ Pipeline Complete! ✨")
            print("="*50)
            print(f"\nSplit dataset location: {split_stats['output_dir']}")
            print(f"Visualization saved to: {viz_path}")
            print("\nYou can now use the split dataset for training your model!")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise


def main():
    """Main entry point"""
    pipeline = TrainingPipeline()
    
    # Run with 80/20 train/val split
    results = pipeline.run_pipeline(
        download=True,  # Set to False to skip download if data exists
        train_ratio=0.8,
        val_ratio=0.2
    )

    
    return results


if __name__ == "__main__":
    results = main()
