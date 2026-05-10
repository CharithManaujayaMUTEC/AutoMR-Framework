import sys
sys.path.append("D:/FYP 78SEm/Datasets")

from load_data import load_images
from tensorflow.keras.models import load_model
from automr.api import AutoMR
import numpy as np

# 🔹 Load dataset
dataset = load_images("D:/FYP 78SEm/Datasets/archive/trafic_data/train/images")

# 🔹 Load pretrained model
model = load_model("D:/FYP 78SEm/Modals/nvidia_model.h5")

# 🔹 Wrap model
class ModelWrapper:
    def predict(self, img):
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return float(model.predict(img)[0])

wrapped_model = ModelWrapper()

# 🔹 AutoMR
automr = AutoMR(wrapped_model)

# 🔹 Run test on 1 image
result = automr.run_all_mrs(dataset[0], samples=20)

print(result.head())