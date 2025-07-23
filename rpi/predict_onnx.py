import onnxruntime as ort
import numpy as np
from PIL import Image
import sys

image_path = sys.argv[1]  # npr. python predict_onnx.py lavirint4.jpg

# Funkcija za transformaciju slike bez torchvision
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.asarray(img).astype(np.float32) / 255.0  # normalizacija
    img_data = np.transpose(img_data, (2, 0, 1))  # [HWC] -> [CHW]
    img_data = np.expand_dims(img_data, axis=0)  # [CHW] -> [NCHW]
    return img_data

# Priprema slike
input_tensor = preprocess_image(image_path)

# Inference preko ONNX
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_tensor})
predicted_class = np.argmax(outputs[0])
print(f"Predviđena klasa: {predicted_class} (slika{predicted_class + 1}.png)")