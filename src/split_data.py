"""
Data splitting module for COCO format datasets
Handles train/validation/test splits while maintaining COCO format
"""

import json
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class COCODataSplitter:
    """Splits COCO format dataset into train/val/test sets"""
    
    def __init__(self, dataset_path: str, seed: int = 42):
        self.dataset_path = Path(dataset_path)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def split_dataset(self, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.2,
                     test_ratio: float = 0.0,
                     output_dir: Optional[str] = None,
                     stratify: bool = True) -> Dict:
        """
        Split COCO dataset into train/val/test sets
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            output_dir: Output directory for split dataset
            stratify: Whether to stratify split by categories
            
        Returns:
            Dictionary with split statistics
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"
        
        # Determine source and output directories
        if output_dir is None:
            output_dir = self.dataset_path.parent / f"{self.dataset_path.name}_split"
        output_dir = Path(output_dir)
        
        # Load original annotations
        original_splits = self._load_all_annotations()
        
        # Combine all data for re-splitting
        combined_data = self._combine_annotations(original_splits)
        
        # Perform split
        if stratify:
            splits = self._stratified_split(combined_data, train_ratio, val_ratio, test_ratio)
        else:
            splits = self._random_split(combined_data, train_ratio, val_ratio, test_ratio)
        
        # Save split datasets
        stats = self._save_splits(splits, output_dir)
        
        return stats
    
    def _load_all_annotations(self) -> Dict:
        """Load all available COCO annotations"""
        annotations = {}
        
        for split in ['train', 'valid', 'test']:
            ann_path = self.dataset_path / split / '_annotations.coco.json'
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    annotations[split] = json.load(f)
                    
        return annotations
    
    def _combine_annotations(self, original_splits: Dict) -> Dict:
        """Combine all splits into a single dataset"""
        combined = {
            'images': [],
            'annotations': [],
            'categories': [],
            'image_paths': {}  # Map image_id to file path
        }
        
        # Get categories (should be same across splits)
        for split_data in original_splits.values():
            if 'categories' in split_data and not combined['categories']:
                combined['categories'] = split_data['categories']
                break
        
        # Combine images and annotations
        image_id_offset = 0
        ann_id_offset = 0
        
        for split_name, split_data in original_splits.items():
            # Create mapping of old to new image IDs
            id_mapping = {}
            
            for img in split_data.get('images', []):
                new_id = img['id'] + image_id_offset
                id_mapping[img['id']] = new_id
                
                new_img = img.copy()
                new_img['id'] = new_id
                combined['images'].append(new_img)
                
                # Store original file path
                combined['image_paths'][new_id] = {
                    'split': split_name,
                    'file_name': img['file_name']
                }
            
            # Update annotations with new IDs
            for ann in split_data.get('annotations', []):
                new_ann = ann.copy()
                new_ann['id'] = ann['id'] + ann_id_offset
                new_ann['image_id'] = id_mapping[ann['image_id']]
                combined['annotations'].append(new_ann)
            
            # Update offsets
            if split_data.get('images'):
                image_id_offset = max(img['id'] for img in combined['images']) + 1
            if split_data.get('annotations'):
                ann_id_offset = max(ann['id'] for ann in combined['annotations']) + 1
                
        return combined
    
    def _stratified_split(self, combined_data: Dict, 
                         train_ratio: float, val_ratio: float, test_ratio: float) -> Dict:
        """Perform stratified split based on category distribution"""
        
        # Group images by their categories
        image_categories = defaultdict(set)
        for ann in combined_data['annotations']:
            image_categories[ann['image_id']].add(ann['category_id'])
        
        # Convert to hashable representation for stratification
        image_ids = list(combined_data['image_paths'].keys())
        image_labels = []
        
        for img_id in image_ids:
            # Use frozenset of categories as label for stratification
            cats = frozenset(image_categories.get(img_id, set()))
            image_labels.append(cats)
        
        # Group images by their category combinations
        label_groups = defaultdict(list)
        for img_id, label in zip(image_ids, image_labels):
            label_groups[label].append(img_id)
        
        # Split each group proportionally
        train_ids, val_ids, test_ids = [], [], []
        
        for label, group_ids in label_groups.items():
            random.shuffle(group_ids)
            n = len(group_ids)
            
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_ids.extend(group_ids[:n_train])
            val_ids.extend(group_ids[n_train:n_train + n_val])
            if test_ratio > 0:
                test_ids.extend(group_ids[n_train + n_val:])
        
        return self._create_splits(combined_data, train_ids, val_ids, test_ids)
    
    def _random_split(self, combined_data: Dict,
                     train_ratio: float, val_ratio: float, test_ratio: float) -> Dict:
        """Perform random split"""
        
        image_ids = list(combined_data['image_paths'].keys())
        random.shuffle(image_ids)
        
        n = len(image_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:] if test_ratio > 0 else []
        
        return self._create_splits(combined_data, train_ids, val_ids, test_ids)
    
    def _create_splits(self, combined_data: Dict,
                      train_ids: List[int], val_ids: List[int], test_ids: List[int]) -> Dict:
        """Create split datasets from image IDs"""
        
        splits = {}
        id_sets = {
            'train': set(train_ids),
            'val': set(val_ids),
            'test': set(test_ids) if test_ids else set()
        }
        
        for split_name, img_id_set in id_sets.items():
            if not img_id_set:
                continue
                
            split_data = {
                'images': [],
                'annotations': [],
                'categories': combined_data['categories'].copy(),
                'info': {
                    'description': f'{split_name} split',
                    'version': '1.0',
                    'year': 2024
                }
            }
            
            # Add images
            for img in combined_data['images']:
                if img['id'] in img_id_set:
                    split_data['images'].append(img)
            
            # Add annotations
            for ann in combined_data['annotations']:
                if ann['image_id'] in img_id_set:
                    split_data['annotations'].append(ann)
            
            # Store original paths
            split_data['_image_paths'] = {
                img_id: combined_data['image_paths'][img_id]
                for img_id in img_id_set
            }
            
            splits[split_name] = split_data
            
        return splits
    
    def _save_splits(self, splits: Dict, output_dir: Path) -> Dict:
        """Save split datasets to disk"""
        
        stats = {
            'output_dir': str(output_dir),
            'splits': {}
        }
        
        for split_name, split_data in splits.items():
            if not split_data['images']:
                continue
                
            # Create split directory
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img in split_data['images']:
                img_info = split_data['_image_paths'][img['id']]
                src_path = self.dataset_path / img_info['split'] / img_info['file_name']
                dst_path = split_dir / img['file_name']
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
            
            # Remove temporary _image_paths before saving
            del split_data['_image_paths']
            
            # Save annotations
            ann_path = split_dir / '_annotations.coco.json'
            with open(ann_path, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            # Collect statistics
            stats['splits'][split_name] = {
                'num_images': len(split_data['images']),
                'num_annotations': len(split_data['annotations']),
                'num_categories': len(split_data['categories']),
                'categories': [cat['name'] for cat in split_data['categories']]
            }
            
            # Category distribution
            cat_counts = defaultdict(int)
            for ann in split_data['annotations']:
                cat_counts[ann['category_id']] += 1
            
            cat_names = {cat['id']: cat['name'] for cat in split_data['categories']}
            stats['splits'][split_name]['category_distribution'] = {
                cat_names[cat_id]: count 
                for cat_id, count in cat_counts.items()
            }
            
        return stats


def split_coco_dataset(dataset_path: str,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.2,
                       test_ratio: float = 0.0,
                       output_dir: Optional[str] = None,
                       stratify: bool = True,
                       seed: int = 42) -> Dict:
    """
    Main function to split COCO dataset
    
    Args:
        dataset_path: Path to the original dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        output_dir: Output directory for split dataset
        stratify: Whether to stratify by categories
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split statistics
    """
    splitter = COCODataSplitter(dataset_path, seed)
    return splitter.split_dataset(train_ratio, val_ratio, test_ratio, output_dir, stratify)


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        stats = split_coco_dataset(dataset_path)
        
        print(f"\n{'='*50}")
        print(f"Dataset Split Complete")
        print(f"{'='*50}")
        print(f"Output: {stats['output_dir']}")
        
        for split_name, split_stats in stats['splits'].items():
            print(f"\n{split_name.upper()}:")
            print(f"  Images: {split_stats['num_images']}")
            print(f"  Annotations: {split_stats['num_annotations']}")
            print(f"  Categories: {split_stats['num_categories']}")
            
            if split_stats.get('category_distribution'):
                print(f"  Category Distribution:")
                for cat_name, count in split_stats['category_distribution'].items():
                    print(f"    {cat_name}: {count}")
