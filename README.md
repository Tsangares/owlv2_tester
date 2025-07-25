# OWLv2 Detection Calibration Environment

Clean testing environment for calibrating OWLv2 truck/trailer detection parameters and vocabulary.

## Quick Start

1. **Run detection calibration:**
   ```bash
   python calibrate_detection.py
   ```

2. **View results:**
   - Annotated images: `test_images/owlv2/annotated_images/`
   - YOLO annotations: `test_images/owlv2/annotations/`  
   - Detection segments: `test_images/owlv2/segments/`
   - Results summary: `test_images/owlv2/calibration_results.json`

## Test Images

Located in `test_images/full/`:
- `sanbernardino_023710134.png` - High-density trailer yard (300+ trailers)
- `losangeles_8009013078.png` - Warehouse loading dock facility
- `losangeles_7001014033.png` - Distribution center with truck parking
- `sanbernardino_023607132.png` - Original trailer yard test case

## Calibration Parameters

Edit `calibrate_detection.py` and modify the `detection_config` dictionary:

### Confidence Thresholds
```python
"confidence_min": 0.005,  # Lower = more detections (try 0.001-0.05)
"confidence_max": 1.0,    # Upper limit (try 0.1-1.0)
```

### Vocabulary Lists
```python
"positive_queries": [
    # Add/remove terms for truck/trailer detection
    "truck", "trailer", "semi-truck", "cargo trailer",
    "trailer parking", "rectangular vehicle", ...
],

"negative_queries": [
    # Add/remove terms to filter false positives  
    "building", "car", "house", "pickup truck", ...
]
```

### Processing Parameters
```python
"nms_threshold": 0.3,     # IoU for duplicate removal (0.1-0.7)
"negative_overlap": 0.3,  # Negative filtering overlap (0.1-0.5)
"max_detections": 200     # Maximum detections per image
```

## Expected Results

### Baseline Performance
- **sanbernardino_023710134**: Should detect 200-300+ trailers
- **losangeles_8009013078**: Should detect 20-30 trucks/trailers
- **losangeles_7001014033**: Should detect 5-15 trucks

### Key Insights
- **Ultra-low thresholds (0.005)** required for trailer detection
- **Comprehensive vocabulary** improves coverage
- **Trailer-specific terms** work better than generic "truck"
- **Visual descriptors** help: "rectangular vehicle", "long vehicle"

## Iteration Workflow

1. **Modify parameters** in `calibrate_detection.py`
2. **Run calibration**: `python calibrate_detection.py`
3. **Check annotated images** in `test_images/owlv2/annotated_images/`
4. **Review detection counts** in terminal output
5. **Adjust and repeat**

## Output Files

### Annotated Images
Visual results with colored bounding boxes:
- Red: High confidence (>0.1)
- Orange: Medium confidence (0.05-0.1)  
- Yellow: Low confidence (0.02-0.05)
- Green: Very low confidence (<0.02)

### YOLO Annotations
Standard YOLO format files for training:
```
class_id x_center y_center width height
```

### Detection Segments
Individual cropped images of each detection for manual review.

### Results Summary
JSON file with:
- Detection counts per image
- Processing statistics
- Configuration used
- Individual detection data

## Tips for Parameter Tuning

### For More Detections
- Lower `confidence_min` (try 0.001)
- Add more vocabulary terms
- Reduce `nms_threshold` (try 0.1-0.2)
- Disable negative filtering

### For Better Precision  
- Raise `confidence_min` (try 0.01-0.05)
- Enable negative filtering
- Increase `nms_threshold` (try 0.5-0.7)
- Add specific negative terms

### For Trailer Yards
- Use ultra-low thresholds (0.005 or lower)
- Include: "trailer", "cargo trailer", "parked trailer"
- Add visual terms: "rectangular vehicle", "colored trailer"

### For Loading Docks
- Moderate thresholds (0.01-0.02)
- Include: "truck at facility", "truck parked"
- Context terms: "delivery truck", "freight truck"

## File Structure
```
src/owl_tester/
├── calibrate_detection.py    # Main calibration script
├── README.md                 # This file
├── test_images/
│   ├── full/                 # Input test images
│   └── owlv2/               # Output results
│       ├── annotated_images/ # Visual results
│       ├── annotations/      # YOLO format
│       ├── segments/         # Individual detections
│       └── calibration_results.json
```