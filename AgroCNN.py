# script_detect.py
import os
import shutil
import math
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.transform import xy
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
from ultralytics import YOLO
from PIL import Image

# ---------------------------
# Helpers
# ---------------------------

from ultralytics import YOLO
import torch

print("CUDA disponível:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

model = YOLO("yolov8n.pt")
model.to("cuda")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_tile_png_from_array(tile_arr, out_path):
    """
    tile_arr: numpy array shape (bands, H, W)
    writes 3-channel PNG using PIL after per-band normalization (1-99 percentiles).
    """
    # ensure at least 3 bands
    bands = tile_arr[:3].astype(np.float32)
    H, W = bands.shape[1], bands.shape[2]
    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(3):
        b = bands[i]
        lo = np.nanpercentile(b, 1)
        hi = np.nanpercentile(b, 99)
        if hi - lo <= 0:
            hi = b.max() if b.max() != b.min() else lo + 1.0
        norm = np.clip((b - lo) / (hi - lo), 0, 1)
        out_img[:, :, i] = (norm * 255).astype(np.uint8)
    Image.fromarray(out_img).save(out_path, format="PNG")

# ---------------------------
# 1) Tile raster (preserve transform & crs)
# ---------------------------
def tile_raster(in_path, out_dir, tile_size=1024, overlap=128):
    ensure_dir(out_dir)
    tiles = []
    with rasterio.open(in_path) as src:
        W, H = src.width, src.height
        step = tile_size - overlap
        idx = 0
        for y in range(0, H, step):
            for x in range(0, W, step):
                w = min(tile_size, W - x)
                h = min(tile_size, H - y)
                win = Window(x, y, w, h)
                transform = rasterio.windows.transform(win, src.transform)
                profile = src.profile.copy()
                profile.update({
                    "height": h,
                    "width": w,
                    "transform": transform
                })
                tile_path = os.path.join(out_dir, f"tile_{idx:05d}.tif")
                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(src.read(window=win))
                tiles.append(tile_path)
                idx += 1
    return tiles

# ---------------------------
# 2) Create YOLO dataset (images + bbox labels)
# ---------------------------
def create_yolo_dataset(tiles_dir, polygons_path, out_dir):
    """
    tiles_dir: folder with tile_XXXXX.tif
    polygons_path: geojson with plant polygons (in same CRS as tiles)
    out_dir: dataset root -> creates images/ and labels/
    returns path to dataset.yaml
    """
    ensure_dir(out_dir)
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    ensure_dir(img_dir); ensure_dir(lbl_dir)

    gdf = gpd.read_file(polygons_path)

    for tile_file in sorted(os.listdir(tiles_dir)):
        if not tile_file.endswith(".tif"):
            continue
        tile_path = os.path.join(tiles_dir, tile_file)
        tile_name = os.path.splitext(tile_file)[0]

        with rasterio.open(tile_path) as src:
            # ensure same CRS
            if gdf.crs != src.crs:
                gdf_tile = gdf.to_crs(src.crs)
            else:
                gdf_tile = gdf

            # filter polygons intersecting tile bounds
            tile_bounds_geom = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            gdf_inter = gdf_tile[gdf_tile.intersects(tile_bounds_geom)]

            # read tile array and save as PNG (normalized)
            tile_arr = src.read()  # (bands, H, W)
            png_path = os.path.join(img_dir, f"{tile_name}.png")
            save_tile_png_from_array(tile_arr, png_path)

            # create label file: one line per polygon intersecting the tile
            label_path = os.path.join(lbl_dir, f"{tile_name}.txt")
            lines = []
            H, W = src.height, src.width
            for _, row in gdf_inter.iterrows():
                # filter by macroclass_id if present
                if 'macroclass_id' in row and row['macroclass_id'] != 1:
                    continue
                minx, miny, maxx, maxy = row.geometry.bounds
                # src.index expects (x,y) and returns (row, col)
                row_ul, col_ul = src.index(minx, maxy)
                row_lr, col_lr = src.index(maxx, miny)
                # clamp
                col_ul = max(0, min(col_ul, W-1))
                col_lr = max(0, min(col_lr, W-1))
                row_ul = max(0, min(row_ul, H-1))
                row_lr = max(0, min(row_lr, H-1))

                x_center = (col_ul + col_lr) / 2.0 / W
                y_center = (row_ul + row_lr) / 2.0 / H
                bw = abs(col_lr - col_ul) / W
                bh = abs(row_lr - row_ul) / H

                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            # write label (can be empty)
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

    # create dataset.yaml
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(img_dir)}\n")
        f.write(f"val: {os.path.abspath(img_dir)}\n")
        f.write("nc: 1\n")
        f.write("names: ['planta']\n")
    return yaml_path

# ---------------------------
# 3) Train YOLOv8 (DETECT)
# ---------------------------
def train_yolo(data_yaml, model_name="yolov8n.pt", imgsz=640, epochs=50, batch=2, device=0):
    """
    Trains YOLOv8 detect model. Defaults tuned for GTX1650.
    """
    model = YOLO(model_name)
    model.train(data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch, device=device)
    return model

# ---------------------------
# 4) Inference (boxes -> geopolygons)
# ---------------------------
def infer_and_merge_boxes(model_path, tiles_dir, ortho_path, out_geojson):
    """
    Runs inference per tile, reads boxes (xyxy), converts to geopolygons (tile CRS), merges all and writes GeoJSON.
    """
    model = YOLO(model_path)
    all_polys = []
    crs = None
    for tile_file in sorted(os.listdir(tiles_dir)):
        if not tile_file.endswith(".tif"):
            continue
        tile_path = os.path.join(tiles_dir, tile_file)
        with rasterio.open(tile_path) as src:
            transform = src.transform
            if crs is None:
                crs = src.crs

        pred = model.predict(tile_path, imgsz=640, verbose=False)[0]

        # boxes in pixels: pred.boxes.xyxy (x1,y1,x2,y2)
        if hasattr(pred, "boxes") and len(pred.boxes) > 0:
            boxes = pred.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                # map pixel coords (col,row) -> geographic using transform
                # use rasterio.transform.xy(row, col)
                # note: xy takes (row, col); here y->row, x->col
                gx1, gy1 = xy(transform, int(y1), int(x1), offset="center")
                gx2, gy2 = xy(transform, int(y2), int(x2), offset="center")
                poly = box(min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2))
                all_polys.append(poly)
        else:
            continue

    gdf = gpd.GeoDataFrame(geometry=all_polys, crs=crs)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf

# ---------------------------
# 5) Export binary raster (plants=1)
# ---------------------------
def export_binary_tif(gdf, ref_raster, out_tif):
    with rasterio.open(ref_raster) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", compress="lzw")
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype='uint8'
        )
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(mask, 1)

# ---------------------------
# 6) Polygons -> points (centroids)
# ---------------------------
def polygons_to_centroids(gdf, out_geojson):
    pts = gdf.copy()
    pts["geometry"] = pts.geometry.centroid
    pts.to_file(out_geojson, driver="GeoJSON")
    return pts

# ---------------------------
# 7) Add NDVI to points
# ---------------------------
def add_ndvi_to_points(points_gdf, raster_path, out_file):
    gdf = points_gdf.copy()
    with rasterio.open(raster_path) as src:
        red = src.read(1).astype(np.float32)
        nir = src.read(3).astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-9)
        vals = []
        for pt in gdf.geometry:
            x, y = pt.x, pt.y
            try:
                row, col = src.index(x, y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    vals.append(float(ndvi[row, col]))
                else:
                    vals.append(None)
            except Exception:
                vals.append(None)
        gdf["NDVI"] = vals
        gdf.to_file(out_file, driver="GeoJSON")
    return gdf

# ---------------------------
# 8) Detection metrics (IoU matching greedy)
# ---------------------------
def evaluate_detection_metrics(gdf_pred, gdf_truth, iou_threshold=0.5):
    preds = list(gdf_pred.geometry)
    truths = list(gdf_truth.geometry)
    if len(preds) == 0 and len(truths) == 0:
        return {"precision":1.0, "recall":1.0, "f1":1.0, "mean_iou":1.0}
    if len(preds) == 0:
        return {"precision":0.0, "recall":0.0, "f1":0.0, "mean_iou":0.0}
    if len(truths) == 0:
        return {"precision":0.0, "recall":0.0, "f1":0.0, "mean_iou":0.0}

    n_pred = len(preds); n_truth = len(truths)
    iou_mat = np.zeros((n_pred, n_truth), dtype=float)
    for i, p in enumerate(preds):
        for j, t in enumerate(truths):
            inter = p.intersection(t).area
            union = p.union(t).area
            iou_mat[i, j] = inter / union if union > 0 else 0.0

    matched_pred = set(); matched_truth = set(); matches = []
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        max_iou = iou_mat[idx]
        if max_iou < iou_threshold or math.isnan(max_iou):
            break
        i, j = idx
        matched_pred.add(i); matched_truth.add(j); matches.append(max_iou)
        iou_mat[i, :] = -1; iou_mat[:, j] = -1

    TP = len(matches); FP = n_pred - TP; FN = n_truth - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matches)) if matches else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "mean_iou": mean_iou, "TP":TP, "FP":FP, "FN":FN, "n_pred":n_pred, "n_truth":n_truth}

# ---------------------------
# MAIN usage (adjust paths)
# ---------------------------
if __name__ == "__main__":
    # === ADJUST PATHS BELOW ===
    ortho_path = r"E:\TRABALHO DE CONCLUSAO DE CURSO\VOOS_OFICIAIS\PLANTIOS\POMAR\ORTHOS\ortho_pomar.tif"
    train_polygons = r"E:\TRABALHO DE CONCLUSAO DE CURSO\VOOS_OFICIAIS\PLANTIOS\POMAR\treino_yolo.geojson"

    base = r"E:\TRABALHO DE CONCLUSAO DE CURSO\VOOS_OFICIAIS\PLANTIOS\POMAR\YOLO"
    tiles_dir = os.path.join(base, "tiles")
    dataset_dir = os.path.join(base, "dataset_yolo")
    output_dir = os.path.join(base, "output")
    ensure_dir(os.path.dirname(tiles_dir)); ensure_dir(dataset_dir); ensure_dir(output_dir)

    output_polys = os.path.join(output_dir, "plantas_detectadas.geojson")
    output_raster = os.path.join(output_dir, "plantas_raster.tif")
    output_points = os.path.join(output_dir, "plantas_centroides.geojson")
    output_points_ndvi = os.path.join(output_dir, "plantas_centroides_ndvi.geojson")

    # 1) tiles
    print("1) Generating tiles...")
    tile_raster(ortho_path, tiles_dir, tile_size=1024, overlap=128)

    # 2) dataset
    print("2) Creating YOLO dataset...")
    data_yaml = create_yolo_dataset(tiles_dir, train_polygons, dataset_dir)
    print(" => dataset.yaml at:", data_yaml)

    # 3) TRAIN (DETECT). Uncomment to run training here (requires GPU)
    print("3) Training YOLO (DETECT) - imgsz=640, batch=2 ...")
    model = train_yolo(data_yaml, model_name="yolov8n.pt", imgsz=640, epochs=50, batch=2, device=0)
    print(" => model trained, best weights at runs/detect/train/weights/best.pt")

    # 4) INFERENCE (use your trained model path)
    model_path = "runs/detect/train/weights/best.pt"  # adjust if different
    print("4) Running inference (boxes) with model:", model_path)
    gdf_pred = infer_and_merge_boxes(model_path, tiles_dir, ortho_path, output_polys)

    # 5) export binary raster
    print("5) Exporting binary raster...")
    export_binary_tif(gdf_pred, ortho_path, output_raster)

    # 6) centroids
    print("6) Creating centroid points...")
    pts = polygons_to_centroids(gdf_pred, output_points)

    # 7) NDVI on centroids
    print("7) Adding NDVI to points...")
    pts_ndvi = add_ndvi_to_points(pts, ortho_path, output_points_ndvi)

    # 8) metrics
    print("8) Evaluating detection metrics...")
    gdf_truth = gpd.read_file(train_polygons)
    metrics = evaluate_detection_metrics(gdf_pred, gdf_truth, iou_threshold=0.5)
    print(metrics)
