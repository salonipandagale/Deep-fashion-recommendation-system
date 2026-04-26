# Deep-fashion-recommendation-system
Deep learning-based fashion recommendation system using VGG16 for feature extraction and cosine similarity for visual search.

# Fashion Recommendation System (CNN-Based)

A deep learning-based fashion recommendation system that suggests visually similar clothing items using image features.

## Overview
This project implements a **content-based recommendation system** using **Convolutional Neural Networks (CNNs)**. It analyzes fashion images and recommends similar items based on visual similarity.

## Methodology
- Used **VGG16 (pre-trained on ImageNet)** for feature extraction
- Removed top classification layers to obtain image embeddings
- Flattened and normalized feature vectors
- Applied **cosine similarity** to measure image similarity
- Retrieved top-N visually similar fashion items
---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Scipy

##  Dataset
- Women Fashion Dataset (image-based)
- Contains various clothing items like dresses, kurtas, etc.
- Link : https://drive.google.com/drive/folders/14TeaFkXLUfBIbVbEVN0L1TesgB0Cn-Mh?usp=sharing

⚠️ Dataset is not included due to size limitations.  
You can use any fashion dataset from Kaggle.

## Output Example
Input image → System recommends visually similar outfits

