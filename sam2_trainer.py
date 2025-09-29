"""
SAM2 (Segment Anything Model 2) Integration Module
For player segmentation and tracking in basketball videos
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict
import pickle


class SAM2Tracker:
    """
    SAM2-based player tracker that combines RT-DETR detections with SAM2 segmentation
    """
    
    def __init__(self, 
                 sam2_checkpoint: str = "sam2_hiera_large.pt",
                 device: Optional[str] = None):
        """
        Initialize SAM2 tracker
        
        Args:
            sam2_checkpoint: Path to SAM2 model checkpoint
            device: Device to use
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam2_checkpoint = sam2_checkpoint
        self.predictor = None
        
        print(f"🎯 Initializing SAM2 Tracker")
        print(f"   Checkpoint: {sam2_checkpoint}")
        print(f"   Device: {self.device}")
        
        self._load_sam2()
    
    def _load_sam2(self):
        """Load SAM2 model"""
        try:
            # Note: This is a placeholder for SAM2 loading
            # Actual implementation depends on SAM2 API when released
            print("⚠️  SAM2 is not publicly released yet.")
            print("   Using SAM2 placeholder. Replace with actual SAM2 when available.")
            print("   For now, using traditional tracking methods.")
            
            # Placeholder - will be replaced with actual SAM2 code:
            # from sam2 import SAM2Predictor
            # self.predictor = SAM2Predictor(self.sam2_checkpoint)
            # self.predictor.to(self.device)
            
            self.predictor = None  # Placeholder
            
        except Exception as e:
            print(f"⚠️  Could not load SAM2: {e}")
            print("   Falling back to traditional tracking")
            self.predictor = None
    
    def track_video(self,
                   video_path: str,
                   detector_model_path: str,
                   output_dir: str = "./output",
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   save_video: bool = True,
                   save_json: bool = True,
                   max_frames: Optional[int] = None) -> Dict:
        """
        Track players in video using RT-DETR + tracking
        
        Args:
            video_path: Path to input video
            detector_model_path: Path to trained RT-DETR model
            output_dir: Output directory
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for tracking
            save_video: Save annotated video
            save_json: Save tracking results as JSON
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            Dictionary with tracking results
        """
        from ultralytics import RTDETR
        
        print("\n" + "="*60)
        print("🎥 Starting Video Tracking")
        print("="*60)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Load detector
        detector = RTDETR(detector_model_path)
        print(f"✅ Loaded detector from: {detector_model_path}")
        
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
        
        print(f"📹 Video Info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        
        # Initialize video writer
        video_writer = None
        if save_video:
            output_video_path = output_path / f"{Path(video_path).stem}_tracked.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
        
        # Tracking data
        tracking_results = {
            'video_path': video_path,
            'fps': fps,
            'resolution': (width, height),
            'frames': [],
            'tracks': defaultdict(list),
        }
        
        # Simple tracker using IoU
        active_tracks = {}
        next_track_id = 0
        
        frame_idx = 0
        print(f"\n🔄 Processing frames...")
        
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = detector.predict(
                frame, 
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Get detections
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            # Track objects
            frame_detections = []
            current_tracks = {}
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                
                # Match with existing tracks
                track_id = self._match_detection_to_track(
                    box, active_tracks, iou_threshold
                )
                
                if track_id is None:
                    # New track
                    track_id = next_track_id
                    next_track_id += 1
                
                # Update track
                current_tracks[track_id] = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class': int(cls),
                    'frame': frame_idx
                }
                
                # Record detection
                detection = {
                    'track_id': track_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'class': int(cls),
                }
                frame_detections.append(detection)
                
                # Add to track history
                tracking_results['tracks'][track_id].append({
                    'frame': frame_idx,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                })
                
                # Draw on frame
                if save_video:
                    # Draw bounding box
                    color = self._get_color(track_id)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                color, 2)
                    
                    # Draw label
                    class_name = results.names[int(cls)]
                    label = f"ID:{track_id} {class_name} {score:.2f}"
                    
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10),
                                (int(x1) + label_w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update active tracks
            active_tracks = current_tracks
            
            # Store frame results
            tracking_results['frames'].append({
                'frame_id': frame_idx,
                'timestamp': frame_idx / fps,
                'detections': frame_detections
            })
            
            # Write frame
            if save_video and video_writer:
                video_writer.write(frame)
            
            # Progress
            if (frame_idx + 1) % 30 == 0 or frame_idx == 0:
                print(f"   Processed: {frame_idx + 1}/{total_frames} frames "
                      f"({(frame_idx + 1) / total_frames * 100:.1f}%)")
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        
        print(f"\n✅ Tracking complete!")
        print(f"   Total frames processed: {frame_idx}")
        print(f"   Total unique tracks: {len(tracking_results['tracks'])}")
        
        # Save JSON results
        if save_json:
            json_path = output_path / f"{Path(video_path).stem}_tracking.json"
            with open(json_path, 'w') as f:
                json.dump(tracking_results, f, indent=2)
            print(f"📄 Tracking data saved to: {json_path}")
        
        if save_video:
            print(f"🎬 Tracked video saved to: {output_video_path}")
        
        return tracking_results
    
    def _match_detection_to_track(self, 
                                  detection_box: np.ndarray,
                                  active_tracks: Dict,
                                  iou_threshold: float = 0.3) -> Optional[int]:
        """
        Match detection to existing track using IoU
        
        Args:
            detection_box: Detection bounding box [x1, y1, x2, y2]
            active_tracks: Dictionary of active tracks
            iou_threshold: IoU threshold for matching
            
        Returns:
            Track ID if matched, None otherwise
        """
        best_iou = 0
        best_track_id = None
        
        for track_id, track_data in active_tracks.items():
            track_box = np.array(track_data['bbox'])
            iou = self._compute_iou(detection_box, track_box)
            
            if iou > best_iou and iou > iou_threshold:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]
        return colors[track_id % len(colors)]
    
    def analyze_tracks(self, tracking_results: Dict) -> Dict:
        """
        Analyze tracking results to extract player statistics
        
        Args:
            tracking_results: Tracking results from track_video
            
        Returns:
            Dictionary with track statistics
        """
        print("\n" + "="*60)
        print("📊 Analyzing Tracks")
        print("="*60)
        
        stats = {
            'total_tracks': len(tracking_results['tracks']),
            'total_frames': len(tracking_results['frames']),
            'tracks_stats': {}
        }
        
        for track_id, track_frames in tracking_results['tracks'].items():
            track_stats = {
                'track_id': track_id,
                'duration_frames': len(track_frames),
                'duration_seconds': len(track_frames) / tracking_results['fps'],
                'avg_confidence': np.mean([f['confidence'] for f in track_frames]),
                'min_confidence': np.min([f['confidence'] for f in track_frames]),
                'max_confidence': np.max([f['confidence'] for f in track_frames]),
                'first_frame': track_frames[0]['frame'],
                'last_frame': track_frames[-1]['frame'],
            }
            
            # Calculate movement
            bboxes = np.array([f['bbox'] for f in track_frames])
            centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
            
            if len(centers) > 1:
                movements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                track_stats['total_movement'] = float(np.sum(movements))
                track_stats['avg_speed'] = float(np.mean(movements))
                track_stats['max_speed'] = float(np.max(movements))
            else:
                track_stats['total_movement'] = 0.0
                track_stats['avg_speed'] = 0.0
                track_stats['max_speed'] = 0.0
            
            stats['tracks_stats'][track_id] = track_stats
        
        # Print summary
        print(f"\n📈 Track Statistics:")
        print(f"   Total tracks: {stats['total_tracks']}")
        print(f"   Total frames: {stats['total_frames']}")
        
        if stats['total_tracks'] > 0:
            durations = [s['duration_seconds'] for s in stats['tracks_stats'].values()]
            print(f"   Avg track duration: {np.mean(durations):.2f}s")
            print(f"   Max track duration: {np.max(durations):.2f}s")
            print(f"   Min track duration: {np.min(durations):.2f}s")
        
        return stats
    
    def visualize_tracks(self, 
                        tracking_results: Dict,
                        output_path: str = "./track_visualization.png"):
        """
        Create visualization of tracking results
        
        Args:
            tracking_results: Tracking results from track_video
            output_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        print(f"\n🎨 Creating track visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Track timeline
        ax1 = axes[0, 0]
        for track_id, track_frames in tracking_results['tracks'].items():
            frames = [f['frame'] for f in track_frames]
            y_pos = [track_id] * len(frames)
            ax1.scatter(frames, y_pos, alpha=0.6, s=2)
        
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Track ID')
        ax1.set_title('Track Timeline', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Detections per frame
        ax2 = axes[0, 1]
        frames = [f['frame_id'] for f in tracking_results['frames']]
        det_counts = [len(f['detections']) for f in tracking_results['frames']]
        ax2.plot(frames, det_counts, linewidth=1)
        ax2.fill_between(frames, det_counts, alpha=0.3)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title('Detections per Frame', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Track duration distribution
        ax3 = axes[1, 0]
        durations = [len(track) / tracking_results['fps'] 
                    for track in tracking_results['tracks'].values()]
        ax3.hist(durations, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Track Duration (seconds)')
        ax3.set_ylabel('Number of Tracks')
        ax3.set_title('Track Duration Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Confidence distribution
        ax4 = axes[1, 1]
        all_confidences = []
        for track in tracking_results['tracks'].values():
            all_confidences.extend([f['confidence'] for f in track])
        
        ax4.hist(all_confidences, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Detection Confidence')
        ax4.set_ylabel('Count')
        ax4.set_title('Confidence Score Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        plt.show()
        
        return output_path


def main():
    """Example usage of SAM2 tracker"""
    
    # Initialize tracker
    tracker = SAM2Tracker()
    
    # Track video
    tracking_results = tracker.track_video(
        video_path="./basketball_game.mp4",
        detector_model_path="./runs/train/basketball_rtdetr/weights/best.pt",
        output_dir="./tracking_output",
        conf_threshold=0.25,
        iou_threshold=0.45,
        save_video=True,
        save_json=True
    )
    
    # Analyze tracks
    stats = tracker.analyze_tracks(tracking_results)
    
    # Visualize
    tracker.visualize_tracks(
        tracking_results,
        output_path="./tracking_output/track_analysis.png"
    )
    
    print("\n" + "="*60)
    print("✨ Tracking Complete!")
    print("="*60)
    
    return tracking_results, stats


if __name__ == "__main__":
    main()