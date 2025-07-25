#!/usr/bin/env python3
"""
OWLv2 Detection Calibration Tool

Clean, focused tool for calibrating detection parameters and vocabulary 
on test images. Run detection, view annotated results, and iterate quickly.

Usage:
    python calibrate_detection.py
    
Results saved to: test_images/owlv2/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionCalibrator:
    def __init__(self):
        """Initialize the detection calibrator."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load OWLv2 model
        logger.info("Loading OWLv2 model...")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
        
        # Test images directory
        self.images_dir = Path("test_images/full")
        self.output_dir = Path("test_images/owlv2")
        
        # Create output directories
        (self.output_dir / "annotated_images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "segments").mkdir(parents=True, exist_ok=True)
        
        # Default detection parameters - MODIFY THESE FOR TESTING
        self.detection_config = {
            # Confidence thresholds
            "confidence_min": 0.005,  # Very low to catch trailers
            "confidence_max": 1.0,    # No upper limit
            
            # Vocabulary lists
            "positive_queries": [
                # Basic terms
                "truck", "semi-truck", "tractor-trailer", "commercial vehicle",
                "trailer", "semi-trailer", "cargo trailer", "box trailer",
                "freight trailer", "truck trailer", "dry van",
                
                # Contextual terms  
                "trailer parking", "trailer in yard", "parked trailer",
                "truck parked", "truck at facility",
                
                # Visual descriptors
                "rectangular vehicle", "long vehicle", "white trailer", 
                "colored trailer", "large vehicle"
            ],
            
            "negative_queries": [
                # Disabled for now - too aggressive
                # "building", "car", "house", "pickup truck", "van"
            ],
            
            # Processing parameters
            "nms_threshold": 0.3,     # IoU threshold for duplicate removal
            "negative_overlap": 0.3,  # Overlap threshold for negative filtering
            "max_detections": 200     # Maximum detections per image
        }

    def detect_with_queries(self, image: Image.Image, queries: List[str], threshold: float) -> List[Dict]:
        """Run OWLv2 detection with specified queries and threshold."""
        try:
            original_size = image.size
            
            inputs = self.processor(text=queries, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([original_size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )
            
            detections = []
            for result in results:
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()
                
                for box, score, label_idx in zip(boxes, scores, labels):
                    if score >= threshold:
                        x1, y1, x2, y2 = box
                        detection = {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(score),
                            "query": queries[label_idx],
                            "query_index": int(label_idx)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def apply_nms(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicates."""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        filtered_detections = []
        for current in detections:
            is_duplicate = False
            for kept in filtered_detections:
                if calculate_iou(current['bbox'], kept['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_detections.append(current)
        
        return filtered_detections

    def enhanced_detection_pipeline(self, image: Image.Image) -> List[Dict]:
        """Run the complete enhanced detection pipeline."""
        config = self.detection_config
        
        # Step 1: Positive detections
        positive_detections = self.detect_with_queries(
            image, 
            config["positive_queries"], 
            config["confidence_min"]
        )
        
        # Step 2: Negative filtering (optional - disable if too aggressive)
        if config["negative_queries"]:
            negative_detections = self.detect_with_queries(
                image, 
                config["negative_queries"], 
                0.1
            )
            
            # Filter overlapping negatives
            filtered_detections = []
            for detection in positive_detections:
                overlaps_negative = False
                for neg_detection in negative_detections:
                    if self.calculate_overlap(detection['bbox'], neg_detection['bbox']) > config["negative_overlap"]:
                        overlaps_negative = True
                        break
                
                if not overlaps_negative:
                    filtered_detections.append(detection)
        else:
            filtered_detections = positive_detections
        
        # Step 3: Confidence range filtering
        confidence_filtered = []
        for detection in filtered_detections:
            if config["confidence_min"] <= detection['confidence'] <= config["confidence_max"]:
                confidence_filtered.append(detection)
        
        # Step 4: Apply NMS
        final_detections = self.apply_nms(confidence_filtered, config["nms_threshold"])
        
        # Step 5: Limit max detections
        if len(final_detections) > config["max_detections"]:
            final_detections = final_detections[:config["max_detections"]]
        
        return final_detections

    def calculate_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Calculate overlap ratio for negative filtering."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        return intersection / area1 if area1 > 0 else 0.0

    def create_annotated_image(self, image_path: Path, detections: List[Dict]) -> str:
        """Create annotated image with detection boxes and labels."""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Color code by confidence level
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            query = detection['query']
            
            # Choose color based on confidence
            if confidence > 0.1:
                color = "red"
            elif confidence > 0.05:
                color = "orange"
            elif confidence > 0.02:
                color = "yellow"
            else:
                color = "green"
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw label
            label = f"{query}: {confidence:.3f}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            
            # Position label above bbox
            label_x = bbox[0]
            label_y = max(0, bbox[1] - label_height - 2)
            
            # Draw label background
            draw.rectangle(
                [label_x, label_y, label_x + label_width, label_y + label_height],
                fill=color
            )
            draw.text((label_x, label_y), label, fill="white", font=font)
        
        # Save annotated image
        output_path = self.output_dir / "annotated_images" / f"{image_path.stem}_annotated.jpg"
        image.save(output_path, "JPEG", quality=95)
        
        return str(output_path)

    def save_yolo_annotations(self, image_path: Path, detections: List[Dict]):
        """Save YOLO format annotations."""
        image = Image.open(image_path)
        img_width, img_height = image.size
        
        annotation_file = self.output_dir / "annotations" / f"{image_path.stem}.txt"
        
        with open(annotation_file, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                
                # Convert to YOLO format (normalized)
                x_center = (bbox[0] + bbox[2]) / 2 / img_width
                y_center = (bbox[1] + bbox[3]) / 2 / img_height
                width = (bbox[2] - bbox[0]) / img_width
                height = (bbox[3] - bbox[1]) / img_height
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def save_detection_segments(self, image_path: Path, detections: List[Dict]):
        """Save individual detection segments as separate images."""
        image = Image.open(image_path).convert("RGB")
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            # Add padding and ensure valid coordinates
            padding = 10
            x1 = max(0, min(bbox[0], bbox[2]) - padding)
            y1 = max(0, min(bbox[1], bbox[3]) - padding)
            x2 = min(image.size[0], max(bbox[0], bbox[2]) + padding)
            y2 = min(image.size[1], max(bbox[1], bbox[3]) + padding)
            
            # Ensure valid crop coordinates
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop segment
            segment = image.crop((x1, y1, x2, y2))
            
            # Save segment
            segment_filename = f"{image_path.stem}_detection_{i:03d}_{detection['confidence']:.3f}.jpg"
            segment_path = self.output_dir / "segments" / segment_filename
            segment.save(segment_path, "JPEG", quality=95)

    def run_calibration(self):
        """Run detection calibration on all test images."""
        image_files = list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg"))
        
        if not image_files:
            logger.error(f"No images found in {self.images_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Detection config: {self.detection_config}")
        
        all_results = []
        total_detections = 0
        start_time = time.time()
        
        for image_path in image_files:
            logger.info(f"Processing: {image_path.name}")
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run detection
            detections = self.enhanced_detection_pipeline(image)
            
            # Save results
            annotated_path = self.create_annotated_image(image_path, detections)
            self.save_yolo_annotations(image_path, detections)
            self.save_detection_segments(image_path, detections)
            
            # Track results
            result = {
                "image_name": image_path.name,
                "total_detections": len(detections),
                "detections": detections,
                "annotated_image": annotated_path
            }
            all_results.append(result)
            total_detections += len(detections)
            
            logger.info(f"  Found {len(detections)} detections")
        
        # Save summary
        elapsed_time = time.time() - start_time
        summary = {
            "detection_config": self.detection_config,
            "total_images": len(image_files),
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / len(image_files),
            "processing_time_seconds": elapsed_time,
            "results": all_results
        }
        
        summary_path = self.output_dir / "calibration_results.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("DETECTION CALIBRATION RESULTS")
        print("="*50)
        print(f"Images processed: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average per image: {total_detections / len(image_files):.1f}")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print(f"\nOutputs saved to: {self.output_dir}")
        print(f"- Annotated images: {self.output_dir}/annotated_images/")
        print(f"- YOLO annotations: {self.output_dir}/annotations/")  
        print(f"- Detection segments: {self.output_dir}/segments/")
        print(f"- Results summary: {summary_path}")
        print("\nModify detection_config in the script to test different parameters!")

def main():
    """Main calibration function."""
    calibrator = DetectionCalibrator()
    calibrator.run_calibration()

if __name__ == "__main__":
    main()