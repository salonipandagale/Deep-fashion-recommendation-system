# -*- coding: utf-8 -*-
"""fashion_recommendation_system_using_image_features.ipynb
"""

import os
import shutil
os.makedirs("images", exist_ok=True)

source_folder = "/content/drive/MyDrive/Deep-fashion-recommendation-system/images"
for file in os.listdir(source_folder):
   if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
      shutil.copy(os.path.join(source_folder, file), "images")

print("Images copied successfully!")

image_directory = "images"

image_paths_list = [
    os.path.join(image_directory, file)
    for file in os.listdir(image_directory)
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
]
# print the list of image file paths
print(image_paths_list)



from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(img_path)


from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from PIL import Image

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]

    # sort in descending order (most similar first)
    similar_indices = np.argsort(similarities)[::-1]

    # remove input image + take top_n
    similar_indices = [
        idx for idx in similar_indices
        if all_image_names[idx] != input_image_path
    ][:top_n]

    plt.figure(figsize=(15, 10))

    # input image
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    # recommendations
    for i, idx in enumerate(similar_indices, start=1):
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(all_image_names[idx]))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#save model
model.save('vgg16_feature_extractor.h5')

"""# You need to give the path of an image as an input, and you will see similar fashion recommendations as output."""

input_image_path = 'images/well-fitted beige suit.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=4)

import pickle

pickle.dump(all_features, open('features.pkl', 'wb'))
pickle.dump(all_image_names, open('filenames.pkl', 'wb'))