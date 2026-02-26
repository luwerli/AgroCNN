# 🍊 Automatic Orange Tree Detection using YOLOv8
**Overview**

This project implements a full geospatial deep learning pipeline for automatic orange tree detection from high-resolution orthomosaics.

It combines computer vision, spatial data engineering, and quantitative evaluation to generate plant-level outputs suitable for Precision Agriculture workflows.

Developed as an undergraduate thesis in Cartographic and Surveying Engineering.

**Problem Statement**

Manual plant counting and vigor assessment in large orchards is:

• Time-consuming

• Error-prone

• Not scalable

This project addresses the problem by integrating object detection with geospatial processing to enable automated, plant-level analysis.

**Solution Architecture**

The pipeline is fully automated and consists of:

**1. Orthomosaic Tiling**

• 1024×1024 tiles

• CRS and affine transform preservation

• Overlap to mitigate edge detection loss

2. Dataset Generation (YOLO Format)

• Conversion of georeferenced polygons into normalized bounding boxes

• Automatic label generation

• dataset.yaml creation

3. Model Training

• YOLOv8n (Ultralytics)

• Image size: 640

• GPU-optimized configuration

**4. Inference & Spatial Reconstruction**

• Tile-based detection

• Bounding box extraction (xyxy format)

• Pixel → geographic coordinate transformation

• GeoJSON export of detected plants

**5. Post-processing**

• Binary raster generation (plants = 1)

• Centroid extraction

• NDVI computation per detected plant

**6. Performance Evaluation**

• Custom IoU-based matching implementation providing:

• Precision

• Recall

• F1-score

• Mean IoU

• TP, FP, FN

Matching is performed using a greedy IoU strategy (threshold ≥ 0.5).

**Technical Highlights**

• End-to-end geospatial Deep Learning pipeline

• Raster ↔ vector conversion with CRS consistency

• Pixel-space to coordinate-space transformation

• Custom detection metrics implementation

• Modular and reproducible workflow


**Tech Stack**

• Python 3.10+

• PyTorch (GPU acceleration)

• Ultralytics YOLOv8

• Rasterio

• GeoPandas

• Shapely

• NumPy

• GDAL

• Matplotlib


**How to Run**
--> Install dependencies
pip install -r requirements.txt

--> Configure input paths
In __main__:

ortho_path = "path/to/orthomosaic.tif"
train_polygons = "path/to/training_data.geojson"

--> Execute full pipeline
python script_detect.py


**Applications**

• Automated orchard inventory

• Detection of planting gaps

• Vegetation vigour monitoring

• Spatial analytics for yield planning

• Precision Agriculture decision support


**Authors**

Luiza Werli Rosa
Thiago Wallace Nascimento da Paz

Geospatial Data Science applied to Agriculture
