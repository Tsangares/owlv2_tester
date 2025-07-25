#!/usr/bin/env python3
"""
OWLv2 Annotation Script for Truck Detection in Satellite Images

This script processes satellite images using OWLv2 (Objectron Localizer with Vision Transformer v2)
to detect trucks and generate:
1. YOLO format annotations (.txt files)
2. Annotated visualization images with bounding boxes
3. Cropped segments for LLaVA validation
4. Metadata mapping for pipeline integration

Usage:
    # Process all images
    python step1_owlvit_annotator.py [--confidence 0.1] [--max-detections 100]
    
    # Process a specific image
    python step1_owlvit_annotator.py --file "image.jpg" [--confidence 0.1] [--max-detections 100]
"""

import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OWLv2TruckAnnotator:
    """
    OWLv2-based truck detection system for satellite imagery annotation
    """
    
    def __init__(self, confidence_threshold: float = 0.1, max_detections: int = 100):
        """
        Initialize the OWLv2 annotator
        
        Args:
            confidence_threshold: Minimum confidence for detections
            max_detections: Maximum number of detections per image
        """
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load OWLv2 model and processor
        self.model_name = "google/owlv2-base-patch16-ensemble"
        logger.info(f"Loading OWLv2 model: {self.model_name}")
        
        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimized truck detection queries based on keyword testing analysis
        # Selected for high precision and good coverage
        self.text_queries = [
            # High precision keywords (>10% validation rate)
            "truck top view",                    # 100% precision (1/1)
            "18 wheeler satellite view",         # 12.8% precision (5/39) 
            "freight truck satellite view",      # 11.8% precision (2/17)
            "truck from above",                  # 7.7% precision (1/13)
            
            # Good coverage keywords (reasonable detection counts)
            "semi truck",
            "truck trailer",
            "tractor-trailer top-down view",
            "semi truck trailer",
            "big rig top view",
            "cargo truck aerial",
            "articulated lorry satellite",
            "tractor trailer aerial view",
            "semi tractor trailer top-down",
            "large commercial vehicle on road",
            "heavy goods vehicle satellite",
            "long haul truck aerial"
        ]
        
        # Output directories
        self.base_dir = Path("/home/wil/desktop/map_cv")
        self.input_dir = self.base_dir / "apple" / "images" / "full"
        self.output_dir = self.base_dir / "apple" / "owlv2"
        self.annotations_dir = self.output_dir / "annotations"
        self.annotated_images_dir = self.output_dir / "annotated_images"
        self.segments_dir = self.output_dir / "segments"
        
        # Create output directories
        for dir_path in [self.annotations_dir, self.annotated_images_dir, self.segments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Metadata for segment tracking
        self.metadata = {
            "detections": [],
            "statistics": {
                "total_images": 0,
                "total_detections": 0,
                "average_confidence": 0.0
            }
        }
        
    def detect_trucks(self, image_path: str) -> List[Dict]:
        """
        Detect trucks in a single image using OWLv2
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Prepare inputs
        inputs = self.processor(
            text=self.text_queries,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process predictions
        target_sizes = torch.tensor([original_size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )
        
        detections = []
        for result in results:
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            
            # Filter and sort by confidence
            valid_indices = scores >= self.confidence_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = labels[valid_indices]
            
            # Sort by confidence (descending) and limit detections
            sorted_indices = np.argsort(scores)[::-1][:self.max_detections]
            
            for idx in sorted_indices:
                bbox = boxes[idx]
                confidence = scores[idx]
                query_idx = labels[idx]
                
                # Convert to normalized YOLO format
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2 / original_size[0]
                y_center = (y1 + y2) / 2 / original_size[1]
                width = (x2 - x1) / original_size[0]
                height = (y2 - y1) / original_size[1]
                
                detection = {
                    "detection_id": str(uuid.uuid4()),
                    "bbox_absolute": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_normalized": [float(x_center), float(y_center), float(width), float(height)],
                    "confidence": float(confidence),
                    "query": self.text_queries[query_idx],
                    "query_index": int(query_idx),
                    "class_id": 0  # All trucks mapped to class 0 for YOLO
                }
                detections.append(detection)
                
        return detections
    
    def save_yolo_annotation(self, detections: List[Dict], image_path: str) -> None:
        """
        Save detections in YOLO format
        
        Args:
            detections: List of detection dictionaries
            image_path: Path to source image
        """
        image_name = Path(image_path).stem
        annotation_file = self.annotations_dir / f"{image_name}.txt"
        
        with open(annotation_file, 'w') as f:
            for detection in detections:
                bbox = detection["bbox_normalized"]
                class_id = detection["class_id"]
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
    def create_annotated_image(self, image_path: str, detections: List[Dict]) -> None:
        """
        Create visualization image with bounding boxes
        
        Args:
            image_path: Path to source image
            detections: List of detection dictionaries
        """
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        # Draw bounding boxes
        for detection in detections:
            bbox = detection["bbox_absolute"]
            confidence = detection["confidence"]
            query = detection["query"]
            
            # Draw rectangle
            draw.rectangle(bbox, outline="red", width=2)
            
            # Draw label
            label = f"{query}: {confidence:.2f}"
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            
            # Position label above bbox
            label_x = bbox[0]
            label_y = max(0, bbox[1] - label_height - 5)
            
            # Draw label background
            draw.rectangle(
                [label_x, label_y, label_x + label_width, label_y + label_height],
                fill="red"
            )
            draw.text((label_x, label_y), label, fill="white", font=font)
            
        # Save annotated image
        image_name = Path(image_path).stem
        output_path = self.annotated_images_dir / f"{image_name}_annotated.jpg"
        image.save(output_path, "JPEG", quality=95)
        
    def save_detection_segments(self, image_path: str, detections: List[Dict]) -> None:
        """
        Save cropped segments for LLaVA validation
        
        Args:
            image_path: Path to source image
            detections: List of detection dictionaries
        """
        image = Image.open(image_path).convert("RGB")
        image_name = Path(image_path).stem
        
        for detection in detections:
            bbox = detection["bbox_absolute"]
            detection_id = detection["detection_id"]
            
            # Add padding around the detection
            padding = 10
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(image.size[0], bbox[2] + padding)
            y2 = min(image.size[1], bbox[3] + padding)
            
            # Crop segment
            segment = image.crop((x1, y1, x2, y2))
            
            # Save segment
            segment_filename = f"{image_name}_{detection_id}.jpg"
            segment_path = self.segments_dir / segment_filename
            segment.save(segment_path, "JPEG", quality=95)
            
            # Update metadata
            self.metadata["detections"].append({
                "detection_id": detection_id,
                "image_path": image_path,
                "segment_path": str(segment_path),
                "bbox_absolute": bbox,
                "bbox_normalized": detection["bbox_normalized"],
                "confidence": detection["confidence"],
                "query": detection["query"],
                "class_id": detection["class_id"]
            })
            
    def save_metadata(self) -> None:
        """
        Save metadata JSON file
        """
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
    def process_single_image(self, filename: str) -> None:
        """
        Process a single image by filename
        
        Args:
            filename: Name of the image file to process
        """
        image_path = self.input_dir / filename
        
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return
            
        logger.info(f"Processing single image: {filename}")
        
        try:
            # Detect trucks
            detections = self.detect_trucks(str(image_path))
            
            if detections:
                # Save YOLO annotations
                self.save_yolo_annotation(detections, str(image_path))
                
                # Create annotated visualization
                self.create_annotated_image(str(image_path), detections)
                
                # Save detection segments
                self.save_detection_segments(str(image_path), detections)
                
                logger.info(f"Found {len(detections)} truck detections")
                
                # Update statistics for single image
                self.metadata["statistics"]["total_images"] = 1
                self.metadata["statistics"]["total_detections"] = len(detections)
                self.metadata["statistics"]["average_confidence"] = (
                    sum(d["confidence"] for d in detections) / len(detections)
                )
            else:
                logger.info("No trucks detected")
                self.metadata["statistics"]["total_images"] = 1
                self.metadata["statistics"]["total_detections"] = 0
                self.metadata["statistics"]["average_confidence"] = 0.0
                
            # Save metadata
            self.save_metadata()
            
            logger.info("Processing complete!")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")

    def process_images(self) -> None:
        """
        Process all images in the input directory
        """
        image_files = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png"))
        total_images = len(image_files)
        total_detections = 0
        confidence_sum = 0.0
        
        logger.info(f"Processing {total_images} images from {self.input_dir}")
        
        for i, image_path in enumerate(image_files, 1):
            try:
                logger.info(f"Processing {i}/{total_images}: {image_path.name}")
                
                # Detect trucks
                detections = self.detect_trucks(str(image_path))
                
                if detections:
                    # Save YOLO annotations
                    self.save_yolo_annotation(detections, str(image_path))
                    
                    # Create annotated visualization
                    self.create_annotated_image(str(image_path), detections)
                    
                    # Save detection segments
                    self.save_detection_segments(str(image_path), detections)
                    
                    # Update statistics
                    total_detections += len(detections)
                    confidence_sum += sum(d["confidence"] for d in detections)
                    
                    logger.info(f"  Found {len(detections)} truck detections")
                else:
                    logger.info("  No trucks detected")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
                
        # Update final statistics
        self.metadata["statistics"]["total_images"] = total_images
        self.metadata["statistics"]["total_detections"] = total_detections
        self.metadata["statistics"]["average_confidence"] = (
            confidence_sum / total_detections if total_detections > 0 else 0.0
        )
        
        # Save metadata
        self.save_metadata()
        
        logger.info(f"Processing complete!")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Average confidence: {self.metadata['statistics']['average_confidence']:.3f}")
        
def main():
    """
    Main function to run the OWLv2 truck annotator
    """
    parser = argparse.ArgumentParser(description="OWLv2 Truck Annotation Pipeline")
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.1, 
        help="Minimum confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "--max-detections", 
        type=int, 
        default=100, 
        help="Maximum detections per image (default: 100)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this specific image file (e.g., 'image.jpg')"
    )
    
    args = parser.parse_args()
    
    # Create annotator
    annotator = OWLv2TruckAnnotator(
        confidence_threshold=args.confidence,
        max_detections=args.max_detections
    )
    
    # Process single file or all images
    if args.file:
        annotator.process_single_image(args.file)
    else:
        annotator.process_images()

if __name__ == "__main__":
    main()