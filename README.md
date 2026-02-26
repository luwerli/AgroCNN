# 🍊 Detecção Automática de Laranjeiras com YOLOv8
📌 Descrição

Projeto de detecção automática de árvores de laranja utilizando YOLOv8 (Ultralytics) aplicado a ortomosaicos de alta resolução.

O pipeline integra:

• Processamento geoespacial

• Deep Learning

• Conversão raster ↔ vetor

• Extração de métricas espectrais

• Avaliação quantitativa de desempenho

• Desenvolvido como Trabalho de Conclusão de Curso em Engenharia Cartográfica.

🎯 Objetivo

Automatizar a:

• Detecção individual de plantas

• Conversão de bounding boxes em geopolígonos

• Geração de raster binário (plantas = 1)

• Extração de NDVI por planta

• Avaliação da performance via IoU

• Aplicação direta em Agricultura de Precisão.

🧠 Arquitetura do Pipeline
1️⃣ Tiling do ortomosaico

Corte em tiles 1024x1024

Preservação de CRS e transform

Overlap para evitar perdas na borda

2️⃣ Criação do Dataset YOLO

Conversão de polígonos para bounding boxes normalizadas

Geração automática de labels

Criação de dataset.yaml

3️⃣ Treinamento

YOLOv8n

imgsz = 640

batch ajustado para GPU GTX 1650

4️⃣ Inferência

Predição por tile

Extração de caixas (xyxy)

Conversão pixel → coordenada geográfica

Geração de GeoJSON

5️⃣ Pós-processamento

Rasterização binária

Cálculo de centróides

Extração de NDVI por planta

6️⃣ Avaliação

Implementação própria de métricas:

• Precision

• Recall

• F1-score

• Mean IoU

• TP, FP, FN

• Matching baseado em IoU com estratégia greedy.

🛠 Tecnologias Utilizadas

• Python 3.10+

• Ultralytics YOLOv8

• PyTorch (GPU)

• Rasterio

• GeoPandas

• Shapely

• NumPy

• GDAL

• Matplotlib
