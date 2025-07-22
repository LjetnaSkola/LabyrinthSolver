import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import sys

image_path = sys.argv[1]  # npr. python predict_onnx.py lavirint4.jpg

# Transformacija kao pri treningu
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Priprema slike
img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()

# Inference preko ONNX
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_tensor})
predicted_class = np.argmax(outputs[0])
print(f"Predviđena klasa: {predicted_class} (slika{predicted_class + 1}.png)")