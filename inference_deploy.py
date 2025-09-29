"""
Real-time Inference and Deployment Utilities
For basketball player detection in images, videos, and live streams
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
from collections import deque
from ultralytics import RTDETR
import json


class BasketballDetector:
    """
    Real-time basketball player detector
    Optimized for inference on images, videos, and webcam streams
    """
    
    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: Optional[str] = None,
                 img_size: int = 640):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to use
            img_size: Input image size
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"🚀 Loading model from: {model_path}")
        self.model = RTDETR(model_path)
        self.model.to(self.device)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        
        print(f"✅ Detector initialized")
        print(f"   Device: {self.device}")
        print(f"   Confidence: {conf_threshold}")
        print(f"   IoU: {iou_threshold}")
    
    def detect_image(self, 
                    image_path: str,
                    save_path: Optional[str] = None,
                    show: bool = False) -> Dict:
        """
        Detect players in a single image
        
        Args:
            image_path: Path to image
            save_path: Path to save annotated image
            show: Display result
            
        Returns:
            Detection results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        start_time = time.time()
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False
        )[0]
        inference_time = time.time() - start_time
        
        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        # Create detection dict
        detections = {
            'image_path': image_path,
            'num_detections': len(boxes),
            'inference_time': inference_time,
            'detections': []
        }
        
        for box, score, cls in zip(boxes, scores, classes):
            detections['detections'].append({
                'bbox': box.tolist(),
                'confidence': float(score),
                'class': results.names[int(cls)],
                'class_id': int(cls)
            })
        
        # Draw annotations
        annotated_img = self._draw_detections(img.copy(), boxes, scores, classes, results.names)
        
        # Add FPS info
        fps_text = f"FPS: {1/inference_time:.1f} | Detections: {len(boxes)}"
        cv2.putText(annotated_img, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"💾 Saved to: {save_path}")
        
        # Show if requested
        if show:
            cv2.imshow('Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
    
    def detect_video(self,
                    video_path: str,
                    output_path: Optional[str] = None,
                    display: bool = False,
                    max_frames: Optional[int] = None,
                    save_json: bool = True) -> Dict:
        """
        Detect players in video
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video
            display: Display video while processing
            max_frames: Maximum frames to process
            save_json: Save detection results as JSON
            
        Returns:
            Detection results for all frames
        """
        print(f"\n🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_results = []
        frame_idx = 0
        
        print("🔄 Processing frames...")
        
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            start_time = time.time()
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
            
            # Calculate FPS
            current_fps = 1 / inference_time if inference_time > 0 else 0
            self.fps_history.append(current_fps)
            avg_fps = np.mean(self.fps_history)
            
            # Extract detections
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Store frame results
            frame_detections = []
            for box, score, cls in zip(boxes, scores, classes):
                frame_detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(score),
                    'class': results.names[int(cls)],
                    'class_id': int(cls)
                })
            
            frame_results.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'num_detections': len(boxes),
                'detections': frame_detections
            })
            
            # Draw annotations
            annotated_frame = self._draw_detections(
                frame.copy(), boxes, scores, classes, results.names
            )
            
            # Add info overlay
            info_text = f"Frame: {frame_idx+1}/{total_frames} | FPS: {avg_fps:.1f} | Detections: {len(boxes)}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display
            if display:
                cv2.imshow('Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if (frame_idx + 1) % 30 == 0:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"   Progress: {frame_idx+1}/{total_frames} ({progress:.1f}%) | "
                      f"Avg FPS: {avg_fps:.1f}")
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Create summary
        results_summary = {
            'video_path': video_path,
            'total_frames': frame_idx,
            'fps': fps,
            'resolution': (width, height),
            'avg_fps': float(np.mean(self.fps_history)) if self.fps_history else 0,
            'avg_detections_per_frame': np.mean([f['num_detections'] for f in frame_results]),
            'frames': frame_results
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   Frames processed: {frame_idx}")
        print(f"   Average FPS: {results_summary['avg_fps']:.1f}")
        print(f"   Avg detections/frame: {results_summary['avg_detections_per_frame']:.1f}")
        
        if output_path:
            print(f"   Saved to: {output_path}")
        
        # Save JSON
        if save_json:
            json_path = Path(video_path).stem + '_detections.json'
            if output_path:
                json_path = str(Path(output_path).parent / json_path)
            
            with open(json_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            print(f"   Detections saved to: {json_path}")
        
        return results_summary
    
    def detect_webcam(self,
                     camera_id: int = 0,
                     save_output: bool = False,
                     output_path: str = "./webcam_output.mp4"):
        """
        Real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
            save_output: Save output video
            output_path: Path to save video
        """
        print(f"\n📹 Starting webcam detection (Camera {camera_id})")
        print("   Press 'q' to quit")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Target FPS
        
        # Setup video writer
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"✅ Camera opened: {width}x{height}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to grab frame")
                break
            
            # Run detection
            start_time = time.time()
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
            
            # Calculate FPS
            current_fps = 1 / inference_time if inference_time > 0 else 0
            self.fps_history.append(current_fps)
            avg_fps = np.mean(self.fps_history)
            
            # Extract detections
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Draw annotations
            annotated_frame = self._draw_detections(
                frame.copy(), boxes, scores, classes, results.names
            )
            
            # Add info overlay
            info_text = f"FPS: {avg_fps:.1f} | Detections: {len(boxes)}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save frame
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display
            cv2.imshow('Webcam Detection', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Webcam detection stopped")
        print(f"   Frames processed: {frame_count}")
        if save_output:
            print(f"   Saved to: {output_path}")
    
    def _draw_detections(self,
                        img: np.ndarray,
                        boxes: np.ndarray,
                        scores: np.ndarray,
                        classes: np.ndarray,
                        class_names: Dict) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        
        colors = {
            0: (0, 255, 0),    # Green
            1: (255, 0, 0),    # Blue
            2: (0, 0, 255),    # Red
            3: (255, 255, 0),  # Cyan
            4: (255, 0, 255),  # Magenta
        }
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_int = int(cls)
            
            # Get color
            color = colors.get(cls_int, (0, 255, 0))
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_names[cls_int]} {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_h - 10),
                        (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    def batch_process_images(self,
                            image_dir: str,
                            output_dir: str,
                            save_json: bool = True) -> Dict:
        """
        Process a directory of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory
            save_json: Save detection results as JSON
            
        Returns:
            Batch processing results
        """
        print(f"\n📁 Batch processing images from: {image_dir}")
        
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_path.glob(ext))
            image_files.extend(image_path.glob(ext.upper()))
        
        print(f"   Found {len(image_files)} images")
        
        # Process each image
        all_results = []
        start_time = time.time()
        
        for i, img_file in enumerate(image_files):
            # Detect
            result = self.detect_image(
                str(img_file),
                save_path=str(output_path / img_file.name),
                show=False
            )
            all_results.append(result)
            
            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Processed: {i+1}/{len(image_files)}")
        
        total_time = time.time() - start_time
        
        # Create summary
        summary = {
            'input_dir': str(image_path),
            'output_dir': str(output_path),
            'num_images': len(image_files),
            'total_time': total_time,
            'avg_time_per_image': total_time / len(image_files) if image_files else 0,
            'results': all_results
        }
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Images processed: {len(image_files)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time/image: {summary['avg_time_per_image']:.2f}s")
        print(f"   Output: {output_path}")
        
        # Save JSON
        if save_json:
            json_path = output_path / 'batch_results.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   Results saved to: {json_path}")
        
        return summary


class DeploymentHelper:
    """
    Helper utilities for model deployment
    """
    
    @staticmethod
    def benchmark_model(model_path: str,
                       img_size: int = 640,
                       num_runs: int = 100) -> Dict:
        """
        Benchmark model performance
        
        Args:
            model_path: Path to model
            img_size: Input image size
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        print(f"\n⚡ Benchmarking model: {model_path}")
        print(f"   Image size: {img_size}")
        print(f"   Runs: {num_runs}")
        
        # Load model
        model = RTDETR(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Create dummy input
        dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        
        # Warmup
        print("   Warming up...")
        for _ in range(10):
            _ = model.predict(dummy_img, verbose=False)
        
        # Benchmark
        print("   Running benchmark...")
        times = []
        
        for i in range(num_runs):
            start = time.time()
            _ = model.predict(dummy_img, verbose=False)
            times.append(time.time() - start)
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{num_runs}")
        
        # Calculate statistics
        times = np.array(times)
        results = {
            'model_path': model_path,
            'device': device,
            'img_size': img_size,
            'num_runs': num_runs,
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'median_time': float(np.median(times)),
            'mean_fps': float(1 / np.mean(times)),
            'p95_time': float(np.percentile(times, 95)),
            'p99_time': float(np.percentile(times, 99)),
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Device: {device}")
        print(f"   Mean inference time: {results['mean_time']*1000:.2f}ms")
        print(f"   Std: {results['std_time']*1000:.2f}ms")
        print(f"   Min: {results['min_time']*1000:.2f}ms")
        print(f"   Max: {results['max_time']*1000:.2f}ms")
        print(f"   Median: {results['median_time']*1000:.2f}ms")
        print(f"   P95: {results['p95_time']*1000:.2f}ms")
        print(f"   P99: {results['p99_time']*1000:.2f}ms")
        print(f"   Mean FPS: {results['mean_fps']:.1f}")
        
        return results
    
    @staticmethod
    def export_for_deployment(model_path: str,
                             output_dir: str = "./deployment",
                             formats: List[str] = ['onnx', 'torchscript']) -> Dict:
        """
        Export model in multiple formats for deployment
        
        Args:
            model_path: Path to trained model
            output_dir: Output directory
            formats: List of export formats
            
        Returns:
            Export results
        """
        print(f"\n📦 Exporting model for deployment")
        print(f"   Model: {model_path}")
        print(f"   Formats: {', '.join(formats)}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        model = RTDETR(model_path)
        
        export_results = {
            'model_path': model_path,
            'output_dir': str(output_path),
            'exports': {}
        }
        
        for fmt in formats:
            print(f"\n   Exporting to {fmt.upper()}...")
            try:
                exported_path = model.export(format=fmt, simplify=True)
                export_results['exports'][fmt] = str(exported_path)
                print(f"   ✅ {fmt.upper()} export successful: {exported_path}")
            except Exception as e:
                print(f"   ❌ {fmt.upper()} export failed: {e}")
                export_results['exports'][fmt] = None
        
        # Save export info
        info_path = output_path / 'export_info.json'
        with open(info_path, 'w') as f:
            json.dump(export_results, f, indent=2)
        
        print(f"\n✅ Export complete!")
        print(f"   Info saved to: {info_path}")
        
        return export_results
    
    @staticmethod
    def create_inference_config(model_path: str,
                               class_names: List[str],
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45,
                               img_size: int = 640,
                               output_path: str = "./inference_config.json"):
        """
        Create inference configuration file
        
        Args:
            model_path: Path to model
            class_names: List of class names
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            img_size: Input image size
            output_path: Path to save config
        """
        config = {
            'model': {
                'path': model_path,
                'type': 'rtdetr',
                'img_size': img_size,
            },
            'inference': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'device': 'auto',  # Auto-detect best device
            },
            'classes': {
                i: name for i, name in enumerate(class_names)
            },
            'preprocessing': {
                'normalize': True,
                'resize': img_size,
            },
            'postprocessing': {
                'nms': True,
                'max_detections': 300,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Inference config saved to: {output_path}")
        
        return config


def main():
    """
    Example usage of inference tools
    """
    
    model_path = "./basketball_pipeline/runs/train/basketball_rtdetr/weights/best.pt"
    
    # Initialize detector
    detector = BasketballDetector(
        model_path=model_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Example 1: Detect in single image
    print("\n" + "="*60)
    print("Example 1: Single Image Detection")
    print("="*60)
    # result = detector.detect_image(
    #     image_path="./test_images/game1.jpg",
    #     save_path="./output/game1_detected.jpg",
    #     show=True
    # )
    
    # Example 2: Process video
    print("\n" + "="*60)
    print("Example 2: Video Processing")
    print("="*60)
    # video_result = detector.detect_video(
    #     video_path="./test_videos/game1.mp4",
    #     output_path="./output/game1_detected.mp4",
    #     display=False,
    #     save_json=True
    # )
    
    # Example 3: Batch process images
    print("\n" + "="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    # batch_result = detector.batch_process_images(
    #     image_dir="./test_images",
    #     output_dir="./output/batch",
    #     save_json=True
    # )
    
    # Example 4: Webcam detection (uncomment to use)
    # print("\n" + "="*60)
    # print("Example 4: Webcam Detection")
    # print("="*60)
    # detector.detect_webcam(
    #     camera_id=0,
    #     save_output=True,
    #     output_path="./output/webcam_recording.mp4"
    # )
    
    # Example 5: Benchmark model
    print("\n" + "="*60)
    print("Example 5: Model Benchmarking")
    print("="*60)
    benchmark_results = DeploymentHelper.benchmark_model(
        model_path=model_path,
        img_size=640,
        num_runs=100
    )
    
    # Example 6: Export for deployment
    print("\n" + "="*60)
    print("Example 6: Export for Deployment")
    print("="*60)
    export_results = DeploymentHelper.export_for_deployment(
        model_path=model_path,
        output_dir="./deployment",
        formats=['onnx']  # Add 'torchscript', 'engine' as needed
    )
    
    # Example 7: Create inference config
    print("\n" + "="*60)
    print("Example 7: Create Inference Config")
    print("="*60)
    config = DeploymentHelper.create_inference_config(
        model_path=model_path,
        class_names=['player', 'ball', 'jersey_number'],  # Update with your classes
        conf_threshold=0.25,
        iou_threshold=0.45,
        img_size=640,
        output_path="./deployment/inference_config.json"
    )
    
    print("\n" + "="*60)
    print("✨ All Examples Complete!")
    print("="*60)
    print("\nUsage Tips:")
    print("1. For real-time performance, use CUDA-enabled GPU")
    print("2. Adjust conf_threshold based on your accuracy/speed needs")
    print("3. Use ONNX format for production deployment")
    print("4. Batch processing is most efficient for multiple images")
    print("5. Monitor FPS to ensure real-time performance")


if __name__ == "__main__":
    main()