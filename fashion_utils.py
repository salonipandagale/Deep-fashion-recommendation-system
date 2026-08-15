import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine


def preprocess_image(img_path):
    img_path = str(img_path).replace("\\", "/")

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return preprocess_input(img_array)

def extract_features(model, img_array):
    features = model.predict(img_array, verbose=0)

    features = features.flatten()

    norm = np.linalg.norm(features)

    if norm == 0:
        return features

    return features / norm


def recommend(image_path, model, features, filenames, top_n=5):
    input_img = preprocess_image(image_path)
    input_features = extract_features(model, input_img)

    similarities = [
        1 - cosine(input_features, f)
        for f in features
    ]

    indices = np.argsort(similarities)[::-1]

    # Normalize paths so they work on both Windows and Linux
    normalized_filenames = [
        str(path).replace("\\", "/")
        for path in filenames
    ]

    normalized_input = str(image_path).replace("\\", "/")

    indices = [
        i for i in indices
        if normalized_filenames[i] != normalized_input
    ][:top_n]

    return [
        normalized_filenames[i]
        for i in indices
    ]
