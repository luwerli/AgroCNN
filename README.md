# AgroCNN: Detecção Automática de Laranjeiras e Análise de Vigor Vegetativo

📌 Descrição

Projeto desenvolvido como Trabalho de Conclusão de Curso com foco na aplicação de Machine Learning e Deep Learning para detecção automática de árvores de laranja a partir de imagens aéreas, além da geração de mapas de vigor vegetal utilizando índices espectrais.

O sistema permite:

• Identificação individualizada de plantas
• Cálculo de NDVI e NDRE por árvore
• Geração de mapas temáticos para suporte à tomada de decisão agrícola

🎯 Problema

A contagem manual e avaliação de vigor de plantas em grandes áreas é:

Demorada;

Sujeita a erro humano;

E pouco escalável

Este projeto busca automatizar:

• Detecção de copas

• Extração de métricas espectrais

• Estruturação de dados por planta

🧠 Metodologia

O pipeline inclui:

• Pré-processamento das imagens aéreas

• Treinamento de modelo de Deep Learning para detecção

• Extração das bounding boxes

• Conversão para geometria espacial

• Cálculo de índices espectrais:
  NDVI
  NDRE

• Geração de shapefile com atributos por planta

• Análise estatística dos resultados

🛠 Tecnologias Utilizadas

• Python

• Pandas

• GeoPandas

• NumPy

• Rasterio

• GDAL

• Matplotlib

• TensorFlow

📊 Resultados

• Detecção automatizada das plantas

• Estruturação de banco espacial por indivíduo

• Mapas de vigor vegetal

• Redução significativa do tempo de análise
