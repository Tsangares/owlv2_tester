#!/usr/bin/env python3
"""
Step 1 (v2): Quad-Process Dual‑GPU OWLv2 Annotation Script — Trailer‑Aware
=========================================================================

This script implements an enhanced OWLv2-based pipeline for detecting
semi-trailers/trailers in large satellite images by:

1. Using a rich set of 60 top-down trailer-specific text queries.
2. Removing any upper-confidence ceiling (keep all boxes above adaptive mins).
3. Loosening negative IoU filtering (threshold=0.55) to avoid dropping half-covered trailers.
4. Splitting the full frame into overlapping 768×768 tiles (stride=512) to boost recall.
5. Applying adaptive minimum thresholds (0.06 for trailer queries, user-specified min otherwise).
6. Falling back to original comprehensive queries when --old-method is passed.

Example:
    python step1_dual_gpu_v2.py images/ --processes 4 --confidence-min 0.12
"""

import os
import json
import uuid
import argparse
import multiprocessing as mp
import logging
import time
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# —────────────────────────────────────────────────────────────────────────────
#  Logger
# —────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# —────────────────────────────────────────────────────────────────────────────
#  Annotator Class
# —────────────────────────────────────────────────────────────────────────────
class OWLv2TruckAnnotator:
    """Enhanced OWLv2 detector for trailers / semi-trucks in satellite imagery."""

    # 60 trailer‑specific, top-down, satellite‑tuned queries
    TRAILER_QUERIES = [
        "semi trailer top view", "semi‑trailer aerial", "detached semi trailer",
        "trailer without tractor top down", "trailer in loading dock",
        "trailer parked in yard", "top‑down cargo trailer",
        "rectangular truck trailer aerial", "white box trailer roof",
        "dry van trailer satellite view", "reefer trailer roof view",
        "shipping container on chassis", "intermodal container trailer top view",
        "40 foot container trailer aerial", "53 foot trailer top view",
        "truck trailer parking lot view", "truck trailer at warehouse",
        "tractorless semi trailer top down", "cargo trailer from above",
        "long rectangular trailer roof", "freight trailer aerial perspective",
        "commercial trailer top view", "parking lane trailer aerial",
        "warehouse docked trailer", "loading bay trailer roof",
        "parking strip trailer top view", "flat roof trailer satellite",
        "grey trailer roof aerial", "silver trailer top down",
        "blue cargo trailer from above", "red container trailer aerial",
        "striped container top view", "bulk trailer roof satellite",
        "yard trailer satellite image", "storage trailer top‑down",
        "parking row trailers aerial", "grid of trailers top view",
        "cargo containers on wheels aerial", "container chassis roof view",
        "top view container trailer cluster", "empty chassis trailer top‑down",
        "semi‑trailer without truck aerial", "trailer queue warehouse",
        "box trailer roof satellite", "roof of cargo trailer",
        "semi dock trailer top view", "fleet of trailers aerial",
        "colored container trailer cluster", "semi‑trailer parking stalls",
        "overhead view freight trailer", "commercial box trailer aerial",
        "long cargo box top‑down", "stacked container trailers",
        "top view rigid trailer", "yard spotted trailer",
        "semi trailer alignment aerial", "trailer bay top‑down",
        "container storage trailers aerial", "lorry trailer roof",
        "warehouse yard trailer aerial"
    ]

    # Negative queries to filter out buildings, cars, houses
    NEGATIVE_QUERIES = ["building", "house", "car"]

    # Original fallback queries for backwards compatibility
    ORIGINAL_QUERIES = [
        "truck top view", "18 wheeler satellite view", "freight truck satellite view",
        "truck from above", "semi truck", "truck trailer",
        "tractor-trailer top-down view", "semi truck trailer", "big rig top view",
        "cargo truck aerial", "articulated lorry satellite", "tractor trailer aerial view",
        "semi tractor trailer top-down", "large commercial vehicle on road",
        "heavy goods vehicle satellite", "long haul truck aerial"
    ]

    def __init__(self,
                 confidence_threshold: float = 0.1,
                 max_detections: int = 50,
                 gpu_id: int = 0,
                 root_dir: str = ".",
                 use_enhanced_method: bool = True):
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.gpu_id = gpu_id
        self.use_enhanced_method = use_enhanced_method

        # Set up device and model
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device).eval()

        # Query list for this run
        self.queries = (self.TRAILER_QUERIES if use_enhanced_method else self.ORIGINAL_QUERIES)

        # Paths
        base = Path(root_dir).expanduser()
        self.images_dir = base / "images" / "full"
        self.annotations_dir = base / "owlv2" / "annotations"
        self.annotated_dir = base / "owlv2" / "annotated_images"
        self.segments_dir = base / "owlv2" / "segments"
        for d in [self.annotations_dir, self.annotated_dir, self.segments_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    #  Helper methods
    # ─────────────────────────────────────────────────────────────────────────
    def _iou(self, b1: List[float], b2: List[float]) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter)

    def _nms(self, dets: List[Dict], iou_thr: float = 0.5) -> List[Dict]:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        keep = []
        for d in dets:
            if all(self._iou(d['bbox_absolute'], k['bbox_absolute']) <= iou_thr for k in keep):
                keep.append(d)
        return keep

    def _detect_with_queries(self, img: Image.Image, queries: List[str], thr: float) -> List[Dict]:
        inputs = self.processor(text=queries, images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outs = self.model(**inputs)
        oh, ow = img.size
        sizes = torch.tensor([[ow, oh]]).to(self.device)
        results = self.processor.post_process_object_detection(outs, threshold=thr, target_sizes=sizes)[0]
        dets = []
        for box, score, lbl in zip(results['boxes'], results['scores'], results['labels']):
            x1, y1, x2, y2 = box.cpu().tolist()
            dets.append({
                'detection_id': str(uuid.uuid4()),
                'bbox_absolute': [x1, y1, x2, y2],
                'confidence': float(score.cpu()),
                'query': queries[int(lbl)],
                'class_id': int(lbl)
            })
        return dets

    def _enhanced_on_image(self, img: Image.Image) -> List[Dict]:
        # positive vs negative
        pos = self._detect_with_queries(img, self.queries, self.confidence_threshold)
        neg = self._detect_with_queries(img, self.NEGATIVE_QUERIES, 0.1)
        filtered = []
        for d in pos:
            if any(self._iou(d['bbox_absolute'], n['bbox_absolute']) > 0.55 for n in neg):
                continue
            low_thr = 0.06 if 'trailer' in d['query'].lower() else self.confidence_threshold
            if d['confidence'] >= low_thr:
                filtered.append(d)
        return self._nms(filtered, 0.5)

    def _detect_tiles(self, img: Image.Image, tile: int = 768, stride: int = 512) -> List[Dict]:
        w, h = img.size
        all_d = []
        for y in range(0, h - tile + 1, stride):
            for x in range(0, w - tile + 1, stride):
                crop = img.crop((x, y, x+tile, y+tile))
                for d in self._enhanced_on_image(crop):
                    x1, y1, x2, y2 = d['bbox_absolute']
                    d['bbox_absolute'] = [x1 + x, y1 + y, x2 + x, y2 + y]
                    all_d.append(d)
        return self._nms(all_d, 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    #  Public detection + utilities
    # ─────────────────────────────────────────────────────────────────────────
    def detect_trucks(self, image_path: Path, image_name: str) -> List[Dict]:
        """
        Main per-image detection entrypoint. Returns list of detections.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            if self.use_enhanced_method:
                dets = self._detect_tiles(img)
            else:
                dets = self._detect_with_queries(img, self.ORIGINAL_QUERIES, self.confidence_threshold)
            for d in dets:
                d['image_path'] = str(image_path)
            if len(dets) > self.max_detections:
                dets = sorted(dets, key=lambda x: x['confidence'], reverse=True)[:self.max_detections]
                logger.info(f"GPU {self.gpu_id}: Limited to {self.max_detections} detections")
            return dets
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Error on {image_name}: {e}")
            return []

    def save_results(self, detections: List[Dict], image_path: Path, image_name: str):
        """
        Save JSON annotations, draw & write annotated image, and crop + save segments.
        """
        # metadata JSON
        md_file = self.annotations_dir / f"{Path(image_name).stem}_detections.json"
        with open(md_file, 'w') as f:
            json.dump(detections, f, indent=2)
        # draw boxes
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for d in detections:
            x1, y1, x2, y2 = d['bbox_absolute']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            label = f"{d['query']}:{d['confidence']:.2f}"
            draw.text((x1, y1-10), label, font=font, fill='red')
        out_img = self.annotated_dir / image_name
        img.save(out_img)
        # crop & save segments
        for d in detections:
            x1, y1, x2, y2 = d['bbox_absolute']
            seg = img.crop((x1, y1, x2, y2))
            seg_file = self.segments_dir / f"{Path(image_name).stem}_{d['detection_id']}.jpg"
            seg.save(seg_file, 'JPEG', quality=90)
            d['segment_path'] = str(seg_file)

    def process_image_batch(self, image_files: List[Path]) -> List[Dict]:
        """Process a list of images on this GPU/process."""
        results = []
        for idx, img_path in enumerate(image_files, 1):
            logger.info(f"GPU {self.gpu_id}: [{idx}/{len(image_files)}] {img_path.name}")
            dets = self.detect_trucks(img_path, img_path.name)
            if dets:
                self.save_results(dets, img_path, img_path.name)
                results.extend(dets)
        return results

# —────────────────────────────────────────────────────────────────────────────
#  Worker & CLI
# —────────────────────────────────────────────────────────────────────────────
def worker_process(gpu_id: int, proc_id: int, batch: List[Path],
                   confidence_min: float, max_detections: int,
                   root_dir: str, result_queue: mp.Queue,
                   use_enhanced: bool):
    annot = OWLv2TruckAnnotator(
        confidence_threshold=confidence_min,
        max_detections=max_detections,
        gpu_id=gpu_id,
        root_dir=root_dir,
        use_enhanced_method=use_enhanced
    )
    start = time.time()
    dets = annot.process_image_batch(batch)
    elapsed = time.time() - start
    logger.info(f"GPU{gpu_id}-P{proc_id} done in {elapsed:.1f}s, {len(dets)} detections")
    result_queue.put(dets)


def main():
    parser = argparse.ArgumentParser(
        description="Dual-GPU OWLv2 Enhanced Trailer Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root_dir", type=str,
                        help="Base folder containing images/full/")
    parser.add_argument("--confidence", type=float, default=0.1,
                        help="Base confidence threshold")
    parser.add_argument("--confidence-min", type=float,
                        help="Override minimum confidence")
    parser.add_argument("--max-detections", type=int, default=50,
                        help="Max detections per image")
    parser.add_argument("--processes", type=int, default=2,
                        help="Total worker processes (will be split 2/GPU)")
    parser.add_argument("--old-method", action="store_true",
                        help="Use original querie set instead of enhanced trailer queries")
    args = parser.parse_args()

    conf_min = args.confidence_min if args.confidence_min is not None else args.confidence
    use_enh = not args.old_method

    base = Path(args.root_dir)
    img_dir = base / "images" / "full"
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not all_imgs:
        logger.error(f"No images found in {img_dir}")
        return

    # Split into 4 roughly equal batches
    num_workers = args.processes
    chunks = [all_imgs[i::num_workers] for i in range(num_workers)]

    mp.set_start_method('spawn')
    q = mp.Queue()
    procs = []
    for i, batch in enumerate(chunks):
        gpu = i % 2  # 0 or 1
        p = mp.Process(target=worker_process,
                       args=(gpu, i, batch, conf_min,
                             args.max_detections, args.root_dir,
                             q, use_enh))
        procs.append(p)
        p.start()

    # Collect results
    all_dets = []
    for _ in procs:
        dets = q.get()
        all_dets.extend(dets)

    for p in procs:
        p.join()

    # Save aggregated metadata
    out_file = base / "owlv2" / "annotations" / "all_detections.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({"detections": all_dets}, f, indent=2)
    logger.info(f"Total detections: {len(all_dets)} saved to {out_file}")

if __name__ == "__main__":
    main()
