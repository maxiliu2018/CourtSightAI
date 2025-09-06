"""
COCO format verification module
Validates that a dataset follows the COCO format structure
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class COCOVerifier:
    """Verifies COCO format dataset structure and content"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.errors = []
        self.warnings = []
        
    def verify(self) -> Tuple[bool, Dict]:
        """
        Perform complete COCO format verification
        Returns: (is_valid, report_dict)
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check directory structure
        if not self._verify_directory_structure():
            report['is_valid'] = False
            report['errors'].extend(self.errors)
            return False, report
            
        # Load and verify annotations
        train_valid = self._verify_annotations('train')
        val_valid = self._verify_annotations('valid')
        test_valid = True
        
        if self.dataset_path / 'test' / '_annotations.coco.json' in self.dataset_path.rglob('*.json'):
            test_valid = self._verify_annotations('test')
        
        # Collect statistics
        stats = self._collect_statistics()
        
        report['is_valid'] = train_valid and val_valid and test_valid
        report['errors'] = self.errors
        report['warnings'] = self.warnings
        report['statistics'] = stats
        
        return report['is_valid'], report
    
    def _verify_directory_structure(self) -> bool:
        """Check if required directories and files exist"""
        required_dirs = ['train', 'valid']
        required_files = {
            'train': '_annotations.coco.json',
            'valid': '_annotations.coco.json'
        }
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self.errors.append(f"Missing required directory: {dir_name}")
                return False
                
            # Check for annotations file
            ann_file = dir_path / required_files[dir_name]
            if not ann_file.exists():
                self.errors.append(f"Missing annotations file: {ann_file}")
                return False
                
            # Check for images
            images = list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpeg'))
            if len(images) == 0:
                self.warnings.append(f"No images found in {dir_name} directory")
                
        return True
    
    def _verify_annotations(self, split: str) -> bool:
        """Verify COCO annotations structure and content"""
        ann_path = self.dataset_path / split / '_annotations.coco.json'
        
        try:
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {ann_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {ann_path}: {e}")
            return False
        
        # Check required top-level keys
        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in coco_data:
                self.errors.append(f"Missing required key '{key}' in {split} annotations")
                return False
        
        # Verify images structure
        if not self._verify_images_structure(coco_data['images'], split):
            return False
            
        # Verify annotations structure
        if not self._verify_annotations_structure(coco_data['annotations'], split):
            return False
            
        # Verify categories structure
        if not self._verify_categories_structure(coco_data['categories'], split):
            return False
            
        # Cross-validate image files exist
        self._validate_image_files_exist(coco_data['images'], split)
        
        return True
    
    def _verify_images_structure(self, images: List[Dict], split: str) -> bool:
        """Verify images array structure"""
        if not isinstance(images, list):
            self.errors.append(f"'images' must be a list in {split}")
            return False
            
        required_fields = ['id', 'file_name', 'height', 'width']
        for idx, img in enumerate(images):
            for field in required_fields:
                if field not in img:
                    self.errors.append(f"Image {idx} missing '{field}' in {split}")
                    return False
                    
        return True
    
    def _verify_annotations_structure(self, annotations: List[Dict], split: str) -> bool:
        """Verify annotations array structure"""
        if not isinstance(annotations, list):
            self.errors.append(f"'annotations' must be a list in {split}")
            return False
            
        required_fields = ['id', 'image_id', 'category_id', 'bbox']
        for idx, ann in enumerate(annotations):
            for field in required_fields:
                if field not in ann:
                    self.errors.append(f"Annotation {idx} missing '{field}' in {split}")
                    return False
                    
            # Verify bbox format [x, y, width, height]
            if not isinstance(ann['bbox'], list) or len(ann['bbox']) != 4:
                self.errors.append(f"Invalid bbox format in annotation {idx} in {split}")
                return False
                
        return True
    
    def _verify_categories_structure(self, categories: List[Dict], split: str) -> bool:
        """Verify categories array structure"""
        if not isinstance(categories, list):
            self.errors.append(f"'categories' must be a list in {split}")
            return False
            
        required_fields = ['id', 'name']
        for idx, cat in enumerate(categories):
            for field in required_fields:
                if field not in cat:
                    self.errors.append(f"Category {idx} missing '{field}' in {split}")
                    return False
                    
        return True
    
    def _validate_image_files_exist(self, images: List[Dict], split: str):
        """Check if image files referenced in annotations actually exist"""
        split_path = self.dataset_path / split
        
        for img in images:
            img_path = split_path / img['file_name']
            if not img_path.exists():
                self.warnings.append(f"Image file not found: {img['file_name']} in {split}")
    
    def _collect_statistics(self) -> Dict:
        """Collect dataset statistics"""
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            ann_path = self.dataset_path / split / '_annotations.coco.json'
            if not ann_path.exists():
                continue
                
            with open(ann_path, 'r') as f:
                coco_data = json.load(f)
                
            split_stats = {
                'num_images': len(coco_data.get('images', [])),
                'num_annotations': len(coco_data.get('annotations', [])),
                'num_categories': len(coco_data.get('categories', [])),
                'categories': [cat['name'] for cat in coco_data.get('categories', [])]
            }
            
            # Count images on disk
            img_path = self.dataset_path / split
            image_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png')) + list(img_path.glob('*.jpeg'))
            split_stats['num_image_files'] = len(image_files)
            
            stats[split] = split_stats
            
        return stats


def verify_coco_dataset(dataset_path: str) -> Tuple[bool, Dict]:
    """
    Main function to verify COCO dataset
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        Tuple of (is_valid, report_dict)
    """
    verifier = COCOVerifier(dataset_path)
    return verifier.verify()


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        is_valid, report = verify_coco_dataset(dataset_path)
        
        print(f"\n{'='*50}")
        print(f"COCO Dataset Verification Report")
        print(f"{'='*50}")
        print(f"Dataset: {dataset_path}")
        print(f"Valid: {is_valid}")
        
        if report['errors']:
            print(f"\nErrors ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"  ❌ {error}")
                
        if report['warnings']:
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"  ⚠️  {warning}")
                
        if report['statistics']:
            print(f"\nStatistics:")
            for split, stats in report['statistics'].items():
                print(f"\n  {split}:")
                print(f"    Images: {stats['num_images']}")
                print(f"    Annotations: {stats['num_annotations']}")
                print(f"    Categories: {stats['num_categories']}")
                print(f"    Image files: {stats['num_image_files']}")
