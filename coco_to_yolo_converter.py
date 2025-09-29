"""
Convert COCO format annotations to YOLO format for Ultralytics training
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All normalized to [0, 1]
    
    Args:
        coco_bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        [x_center, y_center, width, height] normalized
    """
    x, y, w, h = coco_bbox
    
    # Convert to center coordinates
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


def convert_coco_to_yolo(coco_dir, output_dir=None, splits=['train', 'val', 'test']):
    """
    Convert COCO format dataset to YOLO format
    
    Args:
        coco_dir: Directory containing COCO format data
        output_dir: Output directory (defaults to coco_dir + '_yolo')
        splits: List of splits to convert
    """
    coco_path = Path(coco_dir)
    
    if output_dir is None:
        output_dir = str(coco_path) + '_yolo'
    output_path = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"🔄 Converting COCO to YOLO Format")
    print(f"{'='*60}")
    print(f"Input:  {coco_path}")
    print(f"Output: {output_path}")
    
    # Track category mapping
    category_mapping = {}
    
    for split in splits:
        split_dir = coco_path / split
        ann_file = split_dir / '_annotations.coco.json'
        
        if not ann_file.exists():
            print(f"\n⚠️  Skipping {split}: annotation file not found")
            continue
        
        print(f"\n📂 Processing {split} split...")
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create output directories
        output_split_dir = output_path / split
        output_images_dir = output_split_dir / 'images'
        output_labels_dir = output_split_dir / 'labels'
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Build category mapping (COCO ID -> YOLO class index)
        if not category_mapping:
            categories = sorted(coco_data['categories'], key=lambda x: x['id'])
            category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
            category_names = {idx: cat['name'] for idx, cat in enumerate(categories)}
            
            print(f"\n📋 Category Mapping:")
            for idx, name in category_names.items():
                print(f"   Class {idx}: {name}")
        
        # Build image ID to filename mapping
        image_info = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Process each image
        processed_count = 0
        empty_count = 0
        error_count = 0
        
        for img_id, img_data in tqdm(image_info.items(), desc=f"Converting {split}"):
            try:
                img_filename = img_data['file_name']
                img_width = img_data['width']
                img_height = img_data['height']
                
                # Source image path
                src_img_path = split_dir / img_filename
                
                if not src_img_path.exists():
                    error_count += 1
                    continue
                
                # Copy image to output directory
                dst_img_path = output_images_dir / img_filename
                
                # Use symlink or copy based on your preference
                if not dst_img_path.exists():
                    # Copy file
                    import shutil
                    shutil.copy2(src_img_path, dst_img_path)
                
                # Create YOLO label file
                label_filename = Path(img_filename).stem + '.txt'
                label_path = output_labels_dir / label_filename
                
                # Get annotations for this image
                anns = annotations_by_image.get(img_id, [])
                
                if not anns:
                    empty_count += 1
                    # Create empty label file
                    label_path.touch()
                    continue
                
                # Convert annotations to YOLO format
                yolo_lines = []
                for ann in anns:
                    # Get YOLO class index
                    coco_cat_id = ann['category_id']
                    yolo_class_idx = category_mapping[coco_cat_id]
                    
                    # Convert bbox
                    coco_bbox = ann['bbox']
                    yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
                    
                    # Create YOLO line: class x_center y_center width height
                    yolo_line = f"{yolo_class_idx} {' '.join(map(str, yolo_bbox))}"
                    yolo_lines.append(yolo_line)
                
                # Write label file
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n❌ Error processing image {img_id}: {e}")
                error_count += 1
                continue
        
        print(f"\n✅ {split} conversion complete:")
        print(f"   Processed: {processed_count}")
        print(f"   Empty (no annotations): {empty_count}")
        print(f"   Errors: {error_count}")
    
    # Create dataset.yaml
    yaml_content = f"""# YOLO format dataset configuration
path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
"""
    
    for idx, name in sorted(category_names.items()):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_content += f"\nnc: {len(category_names)}  # number of classes\n"
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n{'='*60}")
    print(f"✨ Conversion Complete!")
    print(f"{'='*60}")
    print(f"📁 YOLO dataset location: {output_path}")
    print(f"📄 Dataset config: {yaml_path}")
    print(f"\n🚀 Use this path for training:")
    print(f"   data_yaml='{yaml_path}'")
    
    return str(yaml_path)


def main():
    """Example usage"""
    
    # Convert your split dataset
    yaml_path = convert_coco_to_yolo(
        coco_dir='./basketball_pipeline/data/split_dataset',
        output_dir='./basketball_pipeline/data/split_dataset_yolo',
        splits=['train', 'val']  # Add 'test' if you have it
    )
    
    print(f"\n{'='*60}")
    print("📝 Next Steps:")
    print(f"{'='*60}")
    print("1. Stop the current training (Ctrl+C)")
    print("2. Use the new YOLO format dataset:")
    print(f"\n   from rtdetr_training import RTDETRTrainer")
    print(f"   trainer = RTDETRTrainer()")
    print(f"   results = trainer.train(data_yaml='{yaml_path}', epochs=100)")
    
    return yaml_path


if __name__ == "__main__":
    yaml_path = main()
