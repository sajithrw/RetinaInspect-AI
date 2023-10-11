from tensorflow import keras
from PIL import Image
import numpy as np
import os

# loads saved model
model = keras.models.load_model('DR_detection_model_2.h5')

# loans and preprocesses the test image
image_dir_path = 'test_images/'
image_files = [f for f in os.listdir(image_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

predictions = []
image_filenames = []

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferate DR"]

for image_file in image_files:
    image_path = os.path.join(image_dir_path, image_file)
    image = Image.open(image_path)
    # image = image.resize((224, 224))
    image = np.array(image)

    # normalizes and expands dimensions of the test image
    # image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # makes predictions
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_class_name = class_names[predicted_label[0]]

    predictions.append(predicted_class_name)
    image_filenames.append(image_file)

for filename, label in zip(image_filenames, predictions):
    print(filename.ljust(50) + f" --> {label}")


