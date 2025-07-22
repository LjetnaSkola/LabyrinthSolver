import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import sys


def predict_the_class(image_path, model_path='model.onnx'):
    """
    Funkcija na osnovu fotografije i modela ocijeni koja je klasa originalnog lavirinta
    :param image_path: putanja do fotografije
    :param model_path: istrenirani model
    :return: broj klase od 0 do 5, gje 0 znaci slika1.png, a 5 znači slika6.png
    """

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
    return predicted_class

def main():
    # Putanja do slike i modela
    image_path = sys.argv[1]  # npr. python predict_onnx.py lavirint4.jpg
    predicted_class = predict_the_class(image_path)
    print(f"Predviđena klasa: {predicted_class} (slika{predicted_class + 1}.png)")

if __name__ == "__main__":
    main()